"""Small shared helpers: reproducibility, .mat / image I/O, padding and normalisation."""

import os
import random

import cv2
import numpy as np
import torch
import scipy.io


def set_seed(seed):
    """Make a run reproducible across python, numpy and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class CyclicLoader:
    """Iterate a dataloader forever, restarting it whenever it is exhausted.

    Used to pair the samples of the source and target domains batch by batch.
    """

    def __init__(self, loader):
        self.loader = loader
        self.iterator = iter(loader)

    def next(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            return next(self.iterator)


def load_mat(path, key=None):
    """Load an array from a .mat file (first non-private entry if key is None)."""
    data = scipy.io.loadmat(path)
    if key is not None:
        return np.asarray(data[key], dtype=np.float32)
    for name, value in data.items():
        if not name.startswith('__'):
            return np.asarray(value, dtype=np.float32)
    raise KeyError(f"No array found in {path}")


def load_image(path, colour=True):
    """Read a .bmp / .ppm / .png image as a ``(H, W, C)`` float32 array."""
    flags = cv2.IMREAD_COLOR if colour else cv2.IMREAD_GRAYSCALE
    image = cv2.imread(path, flags)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    if colour:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return np.asarray(image, dtype=np.float32)


def save_mat(path, key, array):
    """Save ``array`` under ``key`` in a .mat file, creating the folder."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    scipy.io.savemat(path, {key: np.asarray(array)})


def pad_to_multiple(img, multiple, is_label=False):
    """Pad the spatial dimensions up to a multiple of ``multiple``."""
    h, w = img.shape[:2]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return img
    if is_label:
        return cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
    if img.ndim == 3:
        return cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
    return np.pad(img, ((0, pad_h), (0, pad_w)), mode='reflect')


def zscore(img):
    """Per-image (per-band) standardisation over the spatial dimensions."""
    img = np.asarray(img, dtype=np.float32)
    mean = img.mean(axis=(0, 1))
    std = img.std(axis=(0, 1))
    if np.isscalar(std):
        std = 1.0 if std == 0 else std
    else:
        std = np.where(std == 0, 1e-8, std)
    return (img - mean) / std


def remap_change_labels(gt, encoding='binary'):
    """Convert the raw change map to the internal convention.

    ``binary``
        ``0 = unchanged``, ``1 = changed``; every pixel is supervised and
        evaluated, which is the encoding of the two released datasets.
    ``ternary``
        ``0 = unlabeled``, ``1 = changed``, ``2 = unchanged``; the unlabeled
        pixels are excluded from training and from the metrics.  Provided for
        partially labelled datasets.

    Returns the binary change map (``0`` unchanged, ``1`` changed) and the mask
    of the evaluated pixels.
    """
    gt = np.asarray(gt)

    if encoding == 'binary':
        change_map = (gt != 0).astype(np.float32)
        labelled = np.ones(gt.shape, dtype=bool)
    elif encoding == 'ternary':
        change_map = (gt == 1).astype(np.float32)
        labelled = gt != 0
    else:
        raise ValueError(f"Unknown change map encoding: {encoding}")

    return change_map, labelled
