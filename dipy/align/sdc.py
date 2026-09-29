"""Fieldmap-less susceptibility distortion correction (SDC).

Estimates the off-resonance displacement of an EPI volume (typically the
mean b=0 of a DWI series) acquired with a single phase-encoding direction,
using an undistorted anatomical image (e.g. T1w) as the geometric reference.

Unlike generic diffeomorphic registration restricted to the phase-encoding
axis (the "SyN-SDC" approach), the displacement is modelled explicitly as the
physics of EPI distortion dictates:

* the displacement is a smooth cubic B-spline field acting only along the
  phase-encoding axis;
* signal pile-up and dilation are accounted for by Jacobian intensity
  modulation, so compressed regions are not penalized for being brighter;
* the anatomical image is mapped to the EPI contrast with a non-parametric
  intensity mapping re-estimated as the correction improves, and compared
  with local normalized cross-correlation;
* the anatomical-to-EPI rigid alignment is re-estimated on the corrected EPI,
  so that distortion does not bias the coregistration.
"""

import numpy as np
from scipy.ndimage import binary_dilation, gaussian_filter, laplace
from scipy.optimize import minimize

from dipy.align.imaffine import (
    AffineRegistration,
    MutualInformationMetric,
    transform_centers_of_mass,
)
from dipy.align.transforms import RigidTransform3D, TranslationTransform3D
from dipy.utils.logging import logger

# ---------------------------------------------------------------------------
# Cubic B-spline displacement model
# ---------------------------------------------------------------------------


def _bspline(t):
    """Cubic B-spline kernel."""
    t = np.abs(t)
    out = np.zeros_like(t)
    m1 = t < 1
    out[m1] = 2.0 / 3.0 - t[m1] ** 2 + 0.5 * t[m1] ** 3
    m2 = (t >= 1) & (t < 2)
    out[m2] = (2.0 - t[m2]) ** 3 / 6.0
    return out


def _bspline_deriv(t):
    """Derivative of the cubic B-spline kernel."""
    a = np.abs(t)
    out = np.zeros_like(t)
    m1 = a < 1
    out[m1] = -2.0 * a[m1] + 1.5 * a[m1] ** 2
    m2 = (a >= 1) & (a < 2)
    out[m2] = -0.5 * (2.0 - a[m2]) ** 2
    return np.sign(t) * out


def _knots(n, spacing):
    """Control point positions (voxel coordinates) covering ``n`` voxels."""
    m = int(np.ceil((n - 1) / spacing)) + 3
    return (n - 1) / 2.0 + (np.arange(m) - (m - 1) / 2.0) * spacing


def _basis(positions, knots, spacing, *, deriv=False):
    t = (positions[:, None] - knots[None, :]) / spacing
    if deriv:
        return _bspline_deriv(t) / spacing
    return _bspline(t)


def _tensor_apply(c, mats):
    """Apply one matrix per axis to a 3D array (separable linear map)."""
    out = np.tensordot(mats[0], c, axes=(1, 0))
    out = np.tensordot(mats[1], out, axes=(1, 1))
    out = np.tensordot(mats[2], out, axes=(1, 2))
    return out.transpose(2, 1, 0)


class _BSplineField:
    """Displacement field (in full-resolution voxels) along one axis."""

    def __init__(self, shape, spacing, pe_axis):
        self.shape = tuple(shape)
        self.spacing = float(spacing)
        self.pe_axis = pe_axis
        self.knots = [_knots(n, self.spacing) for n in self.shape]
        self.cshape = tuple(len(k) for k in self.knots)

    def operators(self, positions):
        """Evaluation and PE-derivative operators at the given sample grid."""
        B = [_basis(p, k, self.spacing) for p, k in zip(positions, self.knots)]
        dB = list(B)
        a = self.pe_axis
        dB[a] = _basis(positions[a], self.knots[a], self.spacing, deriv=True)
        return B, dB

    def fit(self, u, positions):
        """Least-squares coefficients reproducing ``u`` sampled at positions."""
        B, _ = self.operators(positions)
        return _tensor_apply(u, [np.linalg.pinv(b) for b in B])

    def bending(self, c):
        """Discrete bending energy of the coefficients and its gradient."""
        energy = 0.0
        grad = np.zeros_like(c)
        for ax in range(3):
            d2 = np.diff(c, n=2, axis=ax)
            energy += np.sum(d2**2)
            g = np.zeros_like(c)
            sl = [slice(None)] * 3
            for off, w in ((0, 1.0), (1, -2.0), (2, 1.0)):
                sl[ax] = slice(off, off + d2.shape[ax])
                g[tuple(sl)] += 2.0 * w * d2
            grad += g
        for a1, a2 in ((0, 1), (0, 2), (1, 2)):
            d = np.diff(np.diff(c, axis=a1), axis=a2)
            energy += 2.0 * np.sum(d**2)
            g = np.zeros_like(c)
            for o1, w1 in ((0, -1.0), (1, 1.0)):
                for o2, w2 in ((0, -1.0), (1, 1.0)):
                    sl = [slice(None)] * 3
                    sl[a1] = slice(o1, o1 + d.shape[a1])
                    sl[a2] = slice(o2, o2 + d.shape[a2])
                    g[tuple(sl)] += 4.0 * w1 * w2 * d
            grad += g
        return energy, grad


# ---------------------------------------------------------------------------
# Forward model
# ---------------------------------------------------------------------------


def _warp_along_axis(img, disp, axis):
    """Linearly resample ``img`` at ``x + disp * e_axis``.

    Returns the resampled image and the derivative of the interpolant with
    respect to the displacement (in voxels of ``img``).
    """
    im = np.moveaxis(img, axis, -1)
    d = np.moveaxis(disp, axis, -1)
    n = im.shape[-1]
    p = np.arange(n, dtype=np.float64) + d
    inside = (p >= 0) & (p <= n - 1)
    p = np.clip(p, 0, n - 1)
    i0 = np.minimum(np.floor(p).astype(np.intp), n - 2)
    t = p - i0
    v0 = np.take_along_axis(im, i0, axis=-1)
    v1 = np.take_along_axis(im, i0 + 1, axis=-1)
    warped = v0 + t * (v1 - v0)
    slope = np.where(inside, v1 - v0, 0.0)
    return np.moveaxis(warped, -1, axis), np.moveaxis(slope, -1, axis)


def unwarp_epi(data, displacement, pe_axis, *, jacobian=True):
    """Apply an estimated displacement to (possibly 4D) EPI data.

    Parameters
    ----------
    data : ndarray (X, Y, Z) or (X, Y, Z, N)
        Distorted EPI data.
    displacement : ndarray (X, Y, Z)
        Displacement in voxels along ``pe_axis`` defined on the undistorted
        grid, such that ``corrected(x) = data(x + displacement(x) e_pe)``.
    pe_axis : int
        Voxel axis of the phase-encoding direction.
    jacobian : bool, optional
        Modulate intensities by the Jacobian of the displacement.

    Returns
    -------
    corrected : ndarray
        Corrected data with the same shape as ``data``.
    """
    data = np.asarray(data, dtype=np.float64)
    jac = None
    if jacobian:
        jac = np.clip(1.0 + np.gradient(displacement, axis=pe_axis), 0.0, None)
    vols = data[..., None] if data.ndim == 3 else data
    out = np.empty_like(vols)
    for i in range(vols.shape[-1]):
        w, _ = _warp_along_axis(vols[..., i], displacement, pe_axis)
        out[..., i] = w * jac if jacobian else w
    return out[..., 0] if data.ndim == 3 else out


# ---------------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------------


def _lncc(J, S, weights, sigma):
    """Mean local (Gaussian-windowed) correlation and its gradient w.r.t. J."""
    eps = 1e-5

    def G(x):
        return gaussian_filter(x, sigma, mode="constant", truncate=3.0)

    muJ, muS = G(J), G(S)
    A = G(J * S) - muJ * muS
    B = np.maximum(G(J * J) - muJ**2, 0) + eps
    C = np.maximum(G(S * S) - muS**2, 0) + eps
    rsq = np.sqrt(B * C)
    rho = A / rsq
    wsum = weights.sum()
    value = np.sum(weights * rho) / wsum
    a = weights / rsq / wsum
    b = -0.5 * weights * A / (B * rsq) / wsum
    grad = S * G(a) - G(a * muS) + 2.0 * J * G(b) - 2.0 * G(b * muJ)
    return value, grad


def _intensity_map(T, J, mask, *, nbins=64):
    """Map ``T`` to the contrast of ``J`` by the binned conditional mean."""
    t = T[mask]
    j = J[mask]
    lo, hi = np.percentile(t, [0.5, 99.5])
    if hi <= lo:
        return np.full_like(T, j.mean())
    edges = np.linspace(lo, hi, nbins + 1)
    idx = np.clip(np.digitize(t, edges) - 1, 0, nbins - 1)
    counts = np.bincount(idx, minlength=nbins)
    sums = np.bincount(idx, weights=j, minlength=nbins)
    centers = 0.5 * (edges[1:] + edges[:-1])
    ok = counts > 20
    if ok.sum() < 2:
        return np.full_like(T, j.mean())
    means = sums[ok] / counts[ok]
    # light smoothing of the transfer curve
    if means.size > 4:
        means = gaussian_filter(means, 1.0, mode="nearest")
    return np.interp(T, centers[ok], means)


# ---------------------------------------------------------------------------
# Optimization of one resolution level
# ---------------------------------------------------------------------------


class _Level:
    def __init__(
        self,
        field,
        factor,
        epis,
        targets,
        weights,
        channel_weights,
        *,
        sigma,
        lam_bend,
        lam_fold,
        jacobian,
        lam_harm=0.0,
        harm_mask=None,
        prior=None,
        lam_prior=0.0,
    ):
        self.field = field
        self.lam_harm = lam_harm
        self.harm_mask = harm_mask
        self.prior = prior
        self.lam_prior = lam_prior
        self.f = factor
        self.epis = epis
        self.targets = targets
        self.weights = weights
        self.cw = channel_weights
        self.sigma = sigma
        self.lam_bend = lam_bend
        self.lam_fold = lam_fold
        self.jacobian = jacobian
        shape = epis[0].shape
        self.positions = [np.arange(n, dtype=np.float64) * factor for n in shape]
        self.B, self.dB = field.operators(self.positions)
        self.BT = [b.T.copy() for b in self.B]
        self.dBT = [b.T.copy() for b in self.dB]
        self.nvox_ctrl = np.prod(field.cshape)

    def forward(self, c):
        u = _tensor_apply(c, self.B)
        du = _tensor_apply(c, self.dB)
        jac = 1.0 + du
        out = []
        for epi in self.epis:
            w, slope = _warp_along_axis(epi, u / self.f, self.field.pe_axis)
            out.append((w, slope))
        return u, jac, out

    def corrected(self, c):
        _, jac, warped = self.forward(c)
        if self.jacobian:
            return [w * jac for w, _ in warped]
        return [w for w, _ in warped]

    def objective(self, cflat):
        c = cflat.reshape(self.field.cshape)
        u, jac, warped = self.forward(c)
        gu = np.zeros_like(u)
        gdu = np.zeros_like(u)
        loss = 0.0
        for (w, slope), S, cw in zip(warped, self.targets, self.cw):
            J = w * jac if self.jacobian else w
            val, gJ = _lncc(J, S, self.weights, self.sigma)
            loss -= cw * val
            gJ *= -cw
            if self.jacobian:
                gu += gJ * jac * slope / self.f
                gdu += gJ * w
            else:
                gu += gJ * slope / self.f
        # folding penalty on the Jacobian (keeps the mapping invertible)
        thr = 0.1
        viol = np.minimum(jac - thr, 0.0)
        loss += self.lam_fold * np.mean(viol**2)
        gdu += self.lam_fold * 2.0 * viol / viol.size
        if self.lam_harm > 0:
            # the off-resonance field is harmonic where susceptibility is
            # uniform: penalize its Laplacian inside the brain
            hm = self.harm_mask
            Lu = laplace(u, mode="constant") / self.f**2
            nh = max(hm.sum(), 1.0)
            loss += self.lam_harm * np.sum(hm * Lu**2) / nh
            gu += (
                self.lam_harm
                * 2.0
                * laplace(hm * Lu, mode="constant")
                / (self.f**2 * nh)
            )
        if self.lam_prior > 0:
            r = u - self.prior
            nw = self.weights.sum()
            loss += self.lam_prior * np.sum(self.weights * r**2) / nw
            gu += self.lam_prior * 2.0 * self.weights * r / nw
        be, gbe = self.field.bending(c)
        loss += self.lam_bend * be / self.nvox_ctrl
        grad = _tensor_apply(gu, self.BT) + _tensor_apply(gdu, self.dBT)
        grad += self.lam_bend * gbe / self.nvox_ctrl
        return loss, grad.ravel()

    def optimize(self, c0, maxiter):
        res = minimize(
            self.objective,
            c0.ravel(),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": maxiter, "maxcor": 20},
        )
        return res.x.reshape(self.field.cshape), res


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def _normalize(img, mask):
    lo, hi = np.percentile(img[mask], [1, 99])
    return np.clip((img - lo) / max(hi - lo, 1e-12), -0.5, 2.0)


def _downsample(img, factor):
    """Anti-aliased subsampling by an integer factor."""
    if factor == 1:
        return img
    sl = tuple(slice(None, None, factor) for _ in range(img.ndim))
    return gaussian_filter(img, 0.5 * factor)[sl]


def _smooth_to(img, zooms_from, zooms_to):
    """Gaussian anti-aliasing before resampling to a coarser grid."""
    zf = np.asarray(zooms_from, dtype=float)
    zt = np.asarray(zooms_to, dtype=float)
    s = np.sqrt(np.clip(zt**2 - zf**2, 0, None)) / 2.355 / zf
    if np.all(s < 0.1):
        return img
    return gaussian_filter(img, s)


def _rigid(
    static,
    static_affine,
    moving,
    moving_affine,
    *,
    starting_affine=None,
    static_mask=None,
    level_iters=(250, 100, 25),
    sigmas=(3.0, 1.0, 0.0),
    factors=(4, 2, 1),
):
    metric = MutualInformationMetric(nbins=32, sampling_proportion=None)
    reg = AffineRegistration(
        metric=metric,
        level_iters=list(level_iters),
        sigmas=list(sigmas),
        factors=list(factors),
    )
    if starting_affine is None:
        com = transform_centers_of_mass(static, static_affine, moving, moving_affine)
        tr = reg.optimize(
            static,
            moving,
            TranslationTransform3D(),
            None,
            static_grid2world=static_affine,
            moving_grid2world=moving_affine,
            starting_affine=com.affine,
            static_mask=static_mask,
        )
        starting_affine = tr.affine
    return reg.optimize(
        static,
        moving,
        RigidTransform3D(),
        None,
        static_grid2world=static_affine,
        moving_grid2world=moving_affine,
        starting_affine=starting_affine,
        static_mask=static_mask,
    )


class SDCResult:
    """Result of :func:`fieldmapless_sdc`.

    Attributes
    ----------
    displacement : ndarray (X, Y, Z)
        Displacement in voxels along ``pe_axis``, on the EPI grid (undistorted
        space): ``corrected(x) = epi(x + displacement(x) e_pe)``.
    corrected : ndarray (X, Y, Z)
        The corrected EPI reference.
    anat_to_epi : AffineMap
        Rigid mapping between the anatomical image and the corrected EPI.
    fieldmap_hz : ndarray (X, Y, Z) or None
        Off-resonance field in Hz (only if ``readout_time`` was given).
    """

    def __init__(
        self,
        displacement,
        corrected,
        anat_to_epi,
        pe_axis,
        *,
        fieldmap_hz=None,
        jacobian=True,
    ):
        self.displacement = displacement
        self.corrected = corrected
        self.anat_to_epi = anat_to_epi
        self.pe_axis = pe_axis
        self.fieldmap_hz = fieldmap_hz
        self.jacobian = jacobian

    def apply(self, data, *, jacobian=None):
        """Correct EPI data acquired with the same distortion."""
        jac = self.jacobian if jacobian is None else jacobian
        return unwarp_epi(data, self.displacement, self.pe_axis, jacobian=jac)


DEFAULT_SCHEDULE = (
    # (pyramid factor, control spacing [vox], LNCC sigma [level vox], iters)
    (4, 16.0, 1.5, 60),
    (2, 8.0, 1.5, 60),
    (1, 6.0, 1.5, 40),
)


def fieldmapless_sdc(
    epi,
    epi_affine,
    anat,
    anat_affine,
    *,
    pe_axis,
    anat_mask=None,
    epi_extra=None,
    extra_weights=None,
    readout_time=None,
    pe_sign=1,
    schedule=DEFAULT_SCHEDULE,
    n_outer=3,
    n_remap=2,
    lam_bend=0.01,
    lam_fold=10.0,
    lam_harm=1.0,
    prior_fieldmap=None,
    lam_prior=0.03,
    jacobian=True,
    mask_dilation=3,
    anat_to_epi=None,
    verbose=False,
):
    """Fieldmap-less susceptibility distortion correction.

    Parameters
    ----------
    epi : ndarray (X, Y, Z)
        Distorted EPI reference (e.g. mean b0).
    epi_affine : ndarray (4, 4)
        Voxel-to-world affine of ``epi``.
    anat : ndarray
        Undistorted anatomical reference (e.g. skull-stripped T1w).
    anat_affine : ndarray (4, 4)
        Voxel-to-world affine of ``anat``.
    pe_axis : int
        Voxel axis of ``epi`` along which phase encoding was performed.
    anat_mask : ndarray, optional
        Brain mask on the ``anat`` grid. Defaults to ``anat > 0``.
    epi_extra : list of ndarray, optional
        Additional EPI contrasts sharing the distortion of ``epi`` (e.g. the
        mean diffusion-weighted image) used as extra similarity channels.
    extra_weights : list of float, optional
        Weights of the extra channels (the reference channel has weight 1).
    readout_time : float, optional
        Total readout time in seconds, used to report the field in Hz.
    pe_sign : {1, -1}, optional
        Polarity of phase encoding along ``pe_axis``.
    schedule : sequence of tuples, optional
        ``(factor, spacing, sigma, iters)`` per resolution level.
    n_outer : int, optional
        Number of alternations between rigid alignment and field estimation.
    n_remap : int, optional
        Number of intensity-mapping updates per level.
    lam_bend : float, optional
        Weight of the bending-energy regularization.
    lam_fold : float, optional
        Weight of the penalty on Jacobians below 0.1.
    lam_harm : float, optional
        Weight of the penalty on the Laplacian of the displacement inside the
        brain. The off-resonance field is harmonic where the magnetic
        susceptibility is uniform, so this regularizes without opposing the
        strong curvature found near air-tissue interfaces.
    prior_fieldmap : ndarray, optional
        Expected off-resonance field in Hz on the ``anat`` grid (e.g. a
        population-average field warped to the subject). It is mapped to the
        EPI with the current rigid alignment, converted to a displacement
        with ``readout_time`` and ``pe_sign``, and used as initialization and
        as the target of a Tikhonov penalty. Requires ``readout_time``.
    lam_prior : float, optional
        Weight of the penalty towards ``prior_fieldmap``.
    jacobian : bool, optional
        Use Jacobian intensity modulation in the forward model.
    mask_dilation : int, optional
        Dilation (in EPI voxels) of the brain mask defining the similarity
        region, so that the brain boundary contributes.
    anat_to_epi : AffineMap, optional
        Initial rigid alignment (static = EPI, moving = anatomical).
    verbose : bool, optional
        Log progress.

    Returns
    -------
    result : SDCResult
    """
    epi = np.asarray(epi, dtype=np.float64)
    anat = np.asarray(anat, dtype=np.float64)
    if anat_mask is None:
        anat_mask = anat > 0
    anat_mask = np.asarray(anat_mask) > 0
    extras = [np.asarray(e, dtype=np.float64) for e in (epi_extra or [])]
    ew = list(extra_weights) if extra_weights is not None else [0.5] * len(extras)

    zooms_epi = np.sqrt((epi_affine[:3, :3] ** 2).sum(0))
    zooms_anat = np.sqrt((anat_affine[:3, :3] ** 2).sum(0))
    anat_s = _smooth_to(anat, zooms_anat, zooms_epi.mean())
    anat_s = np.where(anat_mask, anat_s, 0.0)

    if prior_fieldmap is not None and readout_time is None:
        raise ValueError("prior_fieldmap requires readout_time")
    field = None
    c = None
    shape = epi.shape
    full_pos = [np.arange(n, dtype=np.float64) for n in shape]
    u = np.zeros(shape)
    u_prior = None
    corrected = epi

    for outer in range(n_outer):
        # -- rigid alignment of the anatomy to the (corrected) EPI -----------
        anat_to_epi = _rigid(
            corrected,
            epi_affine,
            anat_s,
            anat_affine,
            starting_affine=(None if anat_to_epi is None else anat_to_epi.affine),
            level_iters=(250, 100, 25) if outer == 0 else (0, 50, 25),
        )
        T = anat_to_epi.transform(anat_s)
        M = anat_to_epi.transform(anat_mask.astype(np.float64)) > 0.5
        if prior_fieldmap is not None:
            u_prior = (
                pe_sign
                * readout_time
                * anat_to_epi.transform(np.asarray(prior_fieldmap, dtype=np.float64))
            )
            if outer == 0:
                u = u_prior
                c = None
        Md = binary_dilation(M, iterations=mask_dilation)
        channels = [_normalize(epi, M)] + [_normalize(e, M) for e in extras]
        Tn = _normalize(T, M)
        weights_full = Md.astype(np.float64)
        cws = [1.0] + ew

        for f, spacing, sigma, iters in schedule:
            new_field = _BSplineField(shape, spacing, pe_axis)
            if c is None or field is None or field.spacing != new_field.spacing:
                c = new_field.fit(u, full_pos)
            field = new_field

            sl = tuple(slice(None, None, f) for _ in range(3))
            epis_l = [_downsample(ch, f) for ch in channels]
            T_l = _downsample(Tn, f)
            w_l = weights_full[sl]
            m_l = w_l > 0
            hm_l = M[sl].astype(np.float64)
            prior_l = None if u_prior is None else _downsample(u_prior, f)
            for _ in range(n_remap):
                lvl = _Level(
                    field,
                    f,
                    epis_l,
                    [T_l] * len(epis_l),
                    w_l,
                    cws,
                    sigma=sigma,
                    lam_bend=lam_bend,
                    lam_fold=lam_fold,
                    jacobian=jacobian,
                    lam_harm=lam_harm,
                    harm_mask=hm_l,
                    prior=prior_l,
                    lam_prior=lam_prior if prior_l is not None else 0.0,
                )
                cur = lvl.corrected(c)
                targets = [_intensity_map(T_l, J, m_l) for J in cur]
                lvl.targets = targets
                c, res = lvl.optimize(c, max(1, iters // n_remap))
                if verbose:
                    logger.info(
                        f"outer {outer} factor {f} spacing {spacing}: "
                        f"loss {res.fun:.5f} ({res.nit} it)"
                    )
            B, _ = field.operators(full_pos)
            u = _tensor_apply(c, B)
        corrected = unwarp_epi(epi, u, pe_axis, jacobian=jacobian)

    fmap = None
    if readout_time is not None:
        fmap = u / (pe_sign * readout_time)
    return SDCResult(
        u, corrected, anat_to_epi, pe_axis, fieldmap_hz=fmap, jacobian=jacobian
    )
