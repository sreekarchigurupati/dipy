.. _fieldmapless_sdc_flow:

==================================================
Fieldmap-less Susceptibility Distortion Correction
==================================================

EPI acquisitions are geometrically distorted along the phase-encoding (PE)
direction by off-resonance fields, mostly near air-tissue interfaces such as
the sinuses and ear canals. When neither a field map nor a reversed-PE b0 is
available, the distortion can still be estimated by matching the b0 to an
undistorted anatomical image of the same subject.

``dipy_correct_sdc`` estimates a displacement along the PE axis only, modelled
the way EPI distortion behaves physically:

- the displacement is a smooth cubic B-spline field, regularized with its
  bending energy and with its Laplacian inside the brain (the field is
  harmonic where the magnetic susceptibility is uniform);
- intensities are modulated by the Jacobian of the displacement, so signal
  pile-up and stretching are part of the model;
- the anatomical image is mapped to the b0 contrast with a non-parametric
  intensity mapping, re-estimated as the correction improves, and compared
  with local normalized cross-correlation; the mean diffusion-weighted image
  of the lowest shell is used as a second channel;
- the anatomical-to-EPI rigid alignment is re-estimated on the corrected b0,
  so the distortion does not bias the coregistration.

A population-average field map (in Hz, resampled to the subject's anatomical
grid) can optionally be given as a prior; it initializes the estimate and
weakly regularizes it in regions where the images carry little information
(e.g. signal dropout).

The inputs are the (motion-corrected) DWI series, its b-values and
b-vectors, and a skull-stripped T1w image of the same subject::

    dipy_correct_sdc dwi.nii.gz dwi.bval dwi.bvec t1w_brain.nii.gz \
        --pe_direction j- --total_readout 0.05 --out_dir sdc_output

``--pe_direction`` follows the BIDS ``PhaseEncodingDirection`` convention in
voxel axes (``i``, ``i-``, ``j``, ``j-``, ``k``, ``k-``). ``--total_readout``
(``TotalReadoutTime`` in BIDS, seconds) is only needed to write the field map
in Hz and to use ``--prior_fieldmap``. A brain mask on the T1w grid can be
passed with ``--anat_mask``.

The outputs are the corrected series (``dwi_sdc.nii.gz``), the corrected mean
b0 (``b0_sdc.nii.gz``), the displacement in voxels along the PE axis
(``sdc_displacement.nii.gz``) and, if the readout time is known, the field map
in Hz (``sdc_fieldmap_hz.nii.gz``).

The same method is available from Python::

    from dipy.align.sdc import fieldmapless_sdc

    res = fieldmapless_sdc(b0, b0_affine, t1, t1_affine, pe_axis=1,
                           anat_mask=t1_mask, epi_extra=[mean_dwi])
    corrected = res.apply(dwi_data)
