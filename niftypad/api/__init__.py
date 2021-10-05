"""Clean API"""
import logging
from pathlib import Path

from . import readers

log = logging.getLogger(__name__)


def kinetic_model(src, dst=None, params=None, model='srtmb_basis', input_interp_method='linear',
                  w=None, r1=1, k2p=0.000250, beta_lim=None, n_beta=40, linear_phase_start=500,
                  linear_phase_end=None, km_outputs=None, thr=0.1, fig=False):
    """
    Args:
      src (Path or str): input patient directory or filename
      dst (Path or str): output directory (default: `src` directory)
      params (Path or str): config (relative to `src` directory)
      model (str): any model from `niftypad.models` (see `niftypad.models.NAMES`)
      input_interp_method (str): the interpolation method for getting reference input:
        linear, cubic, exp_1, exp_2, feng_srtm
      w (ndarray): weights for weighted model fitting
      r1 (float): a pre-chosen value between 0 and 1 for r1, used in srtmb_asl_basis
      k2p (float): a pre-chosen value for k2p, in second^-1, used in
        srtmb_k2p_basis, logan_ref_k2p, mrtm_k2p
      beta_lim (list[int]): [beta_min, beta_max] for setting the lower and upper limits
        of beta values in basis functions, used in srtmb_basis, srtmb_k2p_basis, srtmb_asl_basis
      n_beta (int): number of beta values/basis functions, used in
        srtmb_basis, srtmb_k2p_basis, srtmb_asl_basis
      linear_phase_start (int): start time of linear phase in seconds, used in logan_ref,
        logan_ref_k2p, mrtm, mrtm_k2p
      linear_phase_end (int): end time of linear phase in seconds, used in logan_ref,
        logan_ref_k2p, mrtm, mrtm_k2p
      km_outputs (list[str]): the kinetic parameters to save, e.g. ['R1', 'k2', 'BP']
      thr (float): threshold value between 0 and 1. Used to mask out voxels with mean value
        over time exceeding `thr * max(image value)`
      fig (bool): whether to show a figure to check model fitting
    """
    import nibabel as nib
    import numpy as np

    from niftypad import basis
    from niftypad.image_process.parametric_image import image_to_parametric
    from niftypad.models import get_model_inputs
    from niftypad.tac import Ref

    src_path = Path(src)
    if src_path.is_dir():
        fpath = next(src_path.glob('*.nii'))
    else:
        fpath = src_path
        src_path = fpath.parent
    log.debug("file:%s", fpath)

    if dst is None:
        dst_path = src_path
    else:
        dst_path = Path(dst)
        assert dst_path.is_dir()

    meta = readers.find_meta(src_path, filter(None, [params, fpath.stem]))
    dt = np.asarray(meta['dt'])
    ref = np.asarray(meta['ref'])
    ref = Ref(ref, dt)
    # change ref interpolation to selected method
    ref.run_interp(input_interp_method=input_interp_method)

    log.debug("looking for first `*.nii` file in %s", src_path)
    img = nib.load(fpath)
    # pet_image = img.get_fdata(dtype=np.float32)
    pet_image = np.asanyarray(img.dataobj)

    # basis functions
    if beta_lim is None:
        beta_lim = [0.01 / 60, 0.3 / 60]
    # change ref.inputf1cubic -> ref.input_interp_1
    b = basis.make_basis(ref.input_interp_1, dt, beta_lim=beta_lim, n_beta=n_beta, w=w, k2p=k2p)

    if km_outputs is None:
        km_outputs = ['R1', 'k2', 'BP']
    # change ref.inputf1cubic -> ref.input_interp_1
    user_inputs = {
        'dt': dt, 'ref': ref, 'inputf1': ref.input_interp_1, 'w': w, 'r1': r1, 'k2p': k2p,
        'beta_lim': beta_lim, 'n_beta': n_beta, 'b': b, 'linear_phase_start': linear_phase_start,
        'linear_phase_end': linear_phase_end, 'fig': fig}
    model_inputs = get_model_inputs(user_inputs, model)
    # log.debug("model_inputs:%s", model_inputs)

    parametric_images_dict, pet_image_fit = image_to_parametric(pet_image, dt, model, model_inputs,
                                                                km_outputs, thr=thr)
    for kp in parametric_images_dict:
        nib.save(nib.Nifti1Image(parametric_images_dict[kp], img.affine),
                 f"{dst_path / fpath.stem}_{model}_{kp}_{fpath.suffix}")
    nib.save(nib.Nifti1Image(pet_image_fit, img.affine),
             f"{dst_path / fpath.stem}_{model}_fit_{fpath.suffix}")
