"""Loading of the raw Shuguang and Island Town datasets.

Each scene is a bi-temporal, bi-modal pair:

* **Shuguang** — ``shuguang_1.bmp`` holds the SAR image of T1 and
  ``shuguang_2.bmp`` the RGB image of T2 (593 x 921 pixels).
* **Island Town** — ``Img11-Ac.ppm`` holds the RGB image of T1 and
  ``Img11-B.ppm`` the SAR image of T2 (403 x 415 pixels).

Both datasets ship full labels: the change map holds ``0 = unchanged`` and
``1 = changed`` and every pixel is evaluated.
"""

import os
from dataclasses import dataclass

import numpy as np

from ..utils import load_image, load_mat, remap_change_labels


@dataclass
class RawDataset:
    """One bi-temporal, bi-modal scene."""

    name: str
    rgb: np.ndarray
    """RGB image, shape ``(H, W, 3)``."""

    sar: np.ndarray
    """SAR image, shape ``(H, W, 3)``."""

    change_map: np.ndarray
    """Binary change map, shape ``(H, W)``, ``1 = changed``."""

    labelled: np.ndarray
    """Boolean mask of the evaluated pixels, shape ``(H, W)``."""

    @property
    def shape(self):
        return self.change_map.shape

    @property
    def num_changed(self):
        return int(np.count_nonzero(self.change_map == 1))

    @property
    def num_unchanged(self):
        return int(np.count_nonzero(self.change_map == 0))


def _read(spec, root, scale, default_loader='mat', is_label=False):
    """Read one entry of a dataset description (image or .mat file)."""
    path = os.path.join(root, spec['file'])
    loader = spec.get('loader', default_loader)

    if loader == 'image':
        value = load_image(path, colour=not is_label)
    elif loader == 'mat':
        value = load_mat(path, spec.get('key'))
    else:
        raise ValueError(f"Unknown loader '{loader}' for {path}")

    if value.ndim == 2 and not is_label:
        value = value[:, :, None]
    elif value.ndim == 3 and is_label:
        value = value[:, :, 0]
    return value.astype(np.float32) * float(scale)


def load_raw_dataset(config, dataset_name):
    """Load the RGB / SAR pair and the change map of ``dataset_name``.

    The file names, the value scaling and the label encoding come from
    ``config['datasets'][dataset_name]``.
    """
    spec = config.get_path(f'datasets.{dataset_name}')
    if spec is None:
        raise KeyError(f"No file description for dataset '{dataset_name}' in the configuration")

    root = config.dataset_dir(dataset_name)
    scale = spec.get('scale', 1.0)
    loader = spec.get('loader', 'mat')

    rgb = _read(spec['rgb'], root, scale, loader)
    sar = _read(spec['sar'], root, scale, loader)
    gt = _read(spec['gt'], root, scale, loader, is_label=True)

    change_map, labelled = remap_change_labels(gt, spec.get('gt_encoding', 'binary'))

    return RawDataset(
        name=dataset_name,
        rgb=np.asarray(rgb, dtype=np.float32),
        sar=np.asarray(sar, dtype=np.float32),
        change_map=change_map,
        labelled=labelled,
    )
