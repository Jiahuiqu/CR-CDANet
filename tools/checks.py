"""Sanity checks of the released implementation.

They run in a few seconds and only need the synthetic data they build
themselves, so no dataset has to be downloaded to run them.

    python tools/checks.py

The model forward pass requires a CUDA device because the selective scan kernel
of ``mamba_ssm`` is GPU only; it is skipped when no GPU is available.
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.metrics import calculate_confusion_matrix, calculate_metrics
from src.refinement.pseudo_label import ConfidenceThresholds, confidence_mask, is_high_confidence


def check_metrics():
    """Compare the metrics against values computed by hand."""
    predictions = np.array([1, 1, 1, 0, 0, 0, 1, 0])
    labels = np.array([1, 1, 0, 0, 0, 1, 1, 0])

    tp, fp, tn, fn = calculate_confusion_matrix(predictions, labels)
    assert (tp, fp, tn, fn) == (3.0, 1.0, 3.0, 1.0), (tp, fp, tn, fn)

    oa, kappa, f1, pr, re = calculate_metrics(predictions, labels)
    assert np.isclose(oa, 6 / 8), oa
    assert np.isclose(pr, 3 / 4), pr
    assert np.isclose(re, 3 / 4), re
    assert np.isclose(f1, 3 / 4), f1
    assert np.isclose(kappa, 0.5), kappa

    print('metrics ..................... ok')


def check_confidence_mask():
    """The vectorised confidence mask must match the per-pixel criterion."""
    rng = np.random.default_rng(0)
    change_map = (rng.random((64, 64)) > 0.5).astype(np.int64)
    thresholds = ConfidenceThresholds()

    for label in (0, 1):
        mask = confidence_mask(change_map, label, thresholds)
        for i in range(change_map.shape[0]):
            for j in range(change_map.shape[1]):
                expected = is_high_confidence(change_map, i, j, label, thresholds)
                assert bool(mask[i, j]) == bool(expected), (i, j, label)

    print('confidence mask equivalence .. ok')


def check_patches():
    """Crops must be centred on their pixel and confined to confident regions."""
    from src.data import ChangePatchDataset, DiffusionPatchDataset

    rng = np.random.default_rng(1)
    height, width, bands = 40, 48, 5
    rgb = rng.random((height, width, 3)).astype(np.float32)
    sar = rng.random((height, width, bands)).astype(np.float32)
    change_map = np.zeros((height, width), dtype=np.float32)

    dataset = ChangePatchDataset([rgb, rgb, sar, sar], change_map, [0, 10, 100], patch_size=17)
    patch_rgb, _, patch_sar, _, label, position = dataset[1]
    assert patch_rgb.shape == (3, 17, 17), patch_rgb.shape
    assert patch_sar.shape == (bands, 17, 17), patch_sar.shape
    assert tuple(position.tolist()) == (0, 10), position
    assert float(label) == 0.0

    diffusion = DiffusionPatchDataset(rgb, sar, change_map, labelled=np.ones_like(change_map, dtype=bool),
                                      crop_size=16, num_samples=8, train_split=1.0)
    assert len(diffusion) == 8
    condition, target = diffusion[0]
    assert condition.shape == (3, 16, 16), condition.shape
    assert target.shape == (bands, 16, 16), target.shape

    print('patch datasets .............. ok')


def check_sampling():
    """The number of sampled points must follow the requested ratios."""
    from src.data import sample_round_points, sample_source_points

    height, width = 60, 60
    change_map = np.zeros((height, width), dtype=np.float32)
    change_map[:10] = 1.0
    labelled = np.ones((height, width), dtype=bool)

    points = sample_source_points(change_map, labelled, 0.1, seed=42)
    assert len(points) == int(600 * 0.1) + int(3000 * 0.1), len(points)

    ratios = [0.01, 0.05, 0.10, 0.15, 0.20]
    for round_index, ratio in enumerate(ratios):
        points = sample_round_points(change_map, change_map, labelled, ratios, round_index, seed=42)
        expected = int(600 * ratio) + int(3000 * ratio)
        assert len(points) == expected, (round_index, len(points), expected)

    print('round sampling ............. ok')


def check_alignment_gradients():
    """The MMD / CORAL loss must back-propagate into the encoders."""
    from src.losses import MultiScaleAlignmentLoss

    torch.manual_seed(0)
    source = [torch.randn(8, 4, 4, 64), torch.randn(8, 2, 2, 128), torch.randn(8, 1, 1, 256)]
    target = [feature.clone().requires_grad_(True) for feature in source]

    _, _, total = MultiScaleAlignmentLoss()(source, target)
    (total + torch.tensor(1.0, requires_grad=True)).backward()

    assert all(feature.grad is not None for feature in target)
    assert any(float(feature.grad.abs().sum()) > 0 for feature in target)

    print('alignment gradient .......... ok')


def check_model_shapes(device='cuda:0'):
    """Forward pass: logit map and encoder feature shapes."""
    from src.models import VMUNet

    patch_size = 17
    model = VMUNet(in_chans_mod1=3, in_chans_mod2=3, num_classes=2).to(device).eval()
    modal1 = [torch.randn(2, 3, patch_size, patch_size).to(device) for _ in range(2)]
    modal2 = [torch.randn(2, 3, patch_size, patch_size).to(device) for _ in range(2)]

    with torch.no_grad():
        logits, *features = model(modal1, modal2)

    assert logits.shape == (2, 2, patch_size - 1, patch_size - 1), logits.shape
    for level, feature in enumerate(features[0]):
        expected = [64 * 4 ** 2, 128 * 2 ** 2, 256]
        assert feature.numel() == 2 * expected[level], (level, feature.shape)

    print(f'model forward ({device}) ......... ok')


def main():
    parser = argparse.ArgumentParser(description="Sanity checks of the CR-CDANet implementation")
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--skip-gpu', action='store_true')
    args = parser.parse_args()

    check_metrics()
    check_confidence_mask()
    check_patches()
    check_sampling()
    check_alignment_gradients()

    if args.skip_gpu or not torch.cuda.is_available():
        print('model forward ............... skipped (no GPU)')
    else:
        check_model_shapes(args.device)

    print('\nAll checks passed.')


if __name__ == '__main__':
    main()
