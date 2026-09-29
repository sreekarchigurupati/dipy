import numpy as np
from numpy.testing import assert_, assert_allclose, assert_array_equal
from scipy.ndimage import gaussian_filter

from dipy.align import sdc


def _phantom(shape=(40, 44, 32), seed=0):
    rng = np.random.default_rng(seed)
    g = np.stack(np.meshgrid(*[np.linspace(-1, 1, n) for n in shape], indexing="ij"))
    r = np.sqrt((g[0] / 0.8) ** 2 + (g[1] / 0.85) ** 2 + (g[2] / 0.8) ** 2)
    brain = r < 1
    tex = gaussian_filter(rng.random(shape), 2.0)
    tex = (tex - tex.min()) / np.ptp(tex)
    wm = brain & (tex > 0.55)
    csf = brain & (tex < 0.35)
    t1 = np.where(brain, 0.6, 0) + 0.35 * wm - 0.45 * csf
    b0 = np.where(brain, 0.7, 0.05) - 0.25 * wm + 0.6 * csf
    return gaussian_filter(t1, 0.6), gaussian_filter(b0, 0.6), brain, g


def _distort(img, u, axis):
    """Inverse of unwarp_epi: produce the EPI that unwarps to ``img``."""
    im = np.moveaxis(img, axis, -1)
    uu = np.moveaxis(u, axis, -1)
    x = np.arange(im.shape[-1])
    du = np.gradient(uu, axis=-1)
    out = np.empty_like(im)
    for idx in np.ndindex(im.shape[:-1]):
        out[idx] = np.interp(x, x + uu[idx], im[idx] / (1 + du[idx]), left=0, right=0)
    return np.moveaxis(out, -1, axis)


def test_unwarp_epi_identity_and_4d():
    rng = np.random.default_rng(1)
    data = rng.random((8, 9, 7, 3))
    corrected = sdc.unwarp_epi(data, np.zeros((8, 9, 7)), 1)
    assert_allclose(corrected, data)
    u = np.full((8, 9, 7), 2.0)
    shifted = sdc.unwarp_epi(data[..., 0], u, 0, jacobian=False)
    assert_allclose(shifted[:6], data[2:, :, :, 0])


def test_unwarp_epi_inverts_distortion():
    _, b0, brain, g = _phantom()
    u = 2.0 * np.exp(-((g[0] - 0.1) ** 2 + (g[1] + 0.4) ** 2 + g[2] ** 2) / 0.2)
    epi = _distort(b0, u, 1)
    corrected = sdc.unwarp_epi(epi, u, 1)
    err = np.abs(corrected - b0)[brain].mean() / b0[brain].mean()
    assert_(err < 0.02)


def test_bspline_fit_reproduces_spline():
    field = sdc._BSplineField((20, 18, 14), 5.0, 0)
    rng = np.random.default_rng(2)
    c = rng.normal(size=field.cshape)
    pos = [np.arange(n, dtype=float) for n in field.shape]
    B, _ = field.operators(pos)
    u = sdc._tensor_apply(c, B)
    c2 = field.fit(u, pos)
    assert_allclose(sdc._tensor_apply(c2, B), u, atol=1e-8)


def test_objective_gradient():
    rng = np.random.default_rng(0)
    shape = (16, 14, 12)
    epi = gaussian_filter(rng.random(shape), 1.5) * 10
    tgt = gaussian_filter(rng.random(shape), 1.5) * 10
    hm = np.zeros(shape)
    hm[3:-3, 3:-3, 3:-3] = 1
    for f in (1, 2):
        field = sdc._BSplineField(tuple(np.array(shape) * f), 5.0, 1)
        lvl = sdc._Level(
            field,
            f,
            [epi],
            [tgt],
            np.ones(shape),
            [1.0],
            sigma=1.5,
            lam_bend=0.3,
            lam_fold=5.0,
            jacobian=True,
            lam_harm=2.0,
            harm_mask=hm,
            prior=rng.normal(size=shape),
            lam_prior=0.5,
        )
        c = rng.normal(0, 0.5, field.cshape).ravel()
        _, grad = lvl.objective(c)
        idx = rng.choice(c.size, 10, replace=False)
        num = np.empty(idx.size)
        for k, i in enumerate(idx):
            e = np.zeros_like(c)
            e[i] = 1e-5
            num[k] = (lvl.objective(c + e)[0] - lvl.objective(c - e)[0]) / 2e-5
        assert_(np.max(np.abs(num - grad[idx])) < 0.02 * np.max(np.abs(num)))


def test_fieldmapless_sdc_recovers_displacement():
    t1, b0, brain, g = _phantom()
    u_true = (
        2.5
        * np.exp(-((g[0] - 0.1) ** 2 + (g[1] + 0.5) ** 2 + (g[2] + 0.3) ** 2) / 0.15)
        - 0.8 * g[2]
    )
    epi = _distort(b0, u_true, 1)
    epi += 0.01 * np.random.default_rng(1).standard_normal(epi.shape)
    aff = np.diag([2.0, 2.0, 2.0, 1.0])
    res = sdc.fieldmapless_sdc(
        epi,
        aff,
        t1,
        aff,
        pe_axis=1,
        anat_mask=brain,
        readout_time=0.05,
        schedule=((2, 8.0, 1.5, 60), (1, 6.0, 1.5, 40)),
    )
    assert_array_equal(res.displacement.shape, epi.shape)
    d = res.displacement - u_true
    d -= np.median(d[brain])
    before = np.abs(u_true - np.median(u_true[brain]))[brain].mean()
    after = np.abs(d[brain]).mean()
    assert_(after < 0.6 * before)
    r_before = np.corrcoef(epi[brain], b0[brain])[0, 1]
    r_after = np.corrcoef(res.corrected[brain], b0[brain])[0, 1]
    assert_(r_after > r_before)
    assert_allclose(res.fieldmap_hz, res.displacement / 0.05)
    assert_allclose(res.apply(epi), res.corrected)


def test_fieldmapless_sdc_prior():
    t1, b0, brain, g = _phantom()
    u_true = 2.5 * np.exp(
        -((g[0] - 0.1) ** 2 + (g[1] + 0.5) ** 2 + (g[2] + 0.3) ** 2) / 0.15
    )
    epi = _distort(b0, u_true, 1)
    aff = np.diag([2.0, 2.0, 2.0, 1.0])
    kw = {
        "pe_axis": 1,
        "anat_mask": brain,
        "schedule": ((2, 8.0, 1.5, 30), (1, 6.0, 1.5, 20)),
        "n_outer": 1,
    }
    with np.testing.assert_raises(ValueError):
        sdc.fieldmapless_sdc(epi, aff, t1, aff, prior_fieldmap=u_true, **kw)
    # a correct prior (in Hz, with pe_sign -1) pins the solution
    readout = 0.05
    prior_hz = -u_true / readout
    res = sdc.fieldmapless_sdc(
        epi,
        aff,
        t1,
        aff,
        prior_fieldmap=prior_hz,
        readout_time=readout,
        pe_sign=-1,
        lam_prior=1.0,
        **kw,
    )
    free = sdc.fieldmapless_sdc(epi, aff, t1, aff, **kw)
    err_prior = np.abs(res.displacement - u_true)[brain].mean()
    err_free = np.abs(free.displacement - u_true)[brain].mean()
    assert_(err_prior < err_free)
    assert_allclose(res.fieldmap_hz, -res.displacement / readout)
