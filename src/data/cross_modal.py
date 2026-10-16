"""Assembly of the four images that the change detection network consumes.

DADiff translates the missing modality of each acquisition, so that every
modality is available at both dates.  The raw data provide the SAR image of one
date and the RGB image of the other, hence the network is fed:

* modality A (RGB) -- ``(RGB synthesized from the SAR image, real RGB image)``
* modality B (SAR) -- ``(SAR synthesized from the RGB image, real SAR image)``

The synthesized images are written by :mod:`train_dadiff` under
``<output_root>/<task>/dadiff/<dataset>/<round>/``.
"""

import os
from dataclasses import dataclass

import numpy as np

from ..utils import load_mat, zscore

GENERATED_RGB = 'rgb_generated.mat'
GENERATED_SAR = 'sar_generated.mat'


@dataclass
class CrossModalPair:
    """The four inputs of one sample."""

    rgb_synthesized: np.ndarray
    rgb_original: np.ndarray
    sar_synthesized: np.ndarray
    sar_original: np.ndarray


def dadiff_dir(config, task_name, dataset_name, round_tag):
    return os.path.join(config.output_root, task_name, 'dadiff', dataset_name, str(round_tag))


def has_dadiff_output(config, task_name, dataset_name, round_tag):
    directory = dadiff_dir(config, task_name, dataset_name, round_tag)
    return (os.path.exists(os.path.join(directory, GENERATED_RGB))
            and os.path.exists(os.path.join(directory, GENERATED_SAR)))


def load_cross_modal(config, raw, task_name, round_tag):
    """Load the synthesized images of ``round_tag`` and pair them with the raw ones.

    Every image is standardised independently, so that the four inputs live on
    the same scale.
    """
    directory = dadiff_dir(config, task_name, raw.name, round_tag)
    if not has_dadiff_output(config, task_name, raw.name, round_tag):
        raise FileNotFoundError(
            f"DADiff output for dataset '{raw.name}' and round '{round_tag}' not found in {directory}. "
            f"Run train_dadiff.py for this dataset first.")

    rgb_synth = load_mat(os.path.join(directory, GENERATED_RGB), 'img')
    sar_synth = load_mat(os.path.join(directory, GENERATED_SAR), 'img')

    if rgb_synth.shape[:2] != raw.rgb.shape[:2]:
        raise ValueError(
            f"Synthesized RGB of '{raw.name}' has shape {rgb_synth.shape}, expected {raw.rgb.shape}")

    return CrossModalPair(
        rgb_synthesized=zscore(rgb_synth),
        rgb_original=zscore(raw.rgb),
        sar_synthesized=zscore(sar_synth),
        sar_original=zscore(raw.sar),
    )
