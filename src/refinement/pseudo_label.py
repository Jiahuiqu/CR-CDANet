"""High-confidence pseudo-label selection.

Candidates are the pixels of the change map produced by the previous refinement
round.  A pixel is accepted when it satisfies either of the two criteria below.

1. *Intra-region scoring*.  For homogeneous regions, the immediate neighbourhood
   (first expansion ring around the pixel, including the pixel) must agree with
   the candidate label at a rate of at least ``tau1`` and the secondary
   neighbourhood (second expansion ring) at a rate of at least ``tau2``.

2. *Boundary scoring*.  For pixels close to a change boundary, the 9x9
   neighbourhood is partitioned into eight directionally staggered sub-regions.
   A pixel is accepted when at least ``min_regions`` sub-regions agree with the
   candidate label at a rate of at least ``tau_b``.

The defaults follow the paper: ``tau1 = 0.8``, ``tau2 = 0.6``, ``tau_b = 0.8``.
"""

from dataclasses import dataclass

import cv2
import numpy as np

# The eight sub-regions of the 9x9 neighbourhood, numbered clockwise from the
# upper-left wedge.  The centre is marked 0 and is excluded from the statistics.
_BOUNDARY_MASK = (
    (1, 1, 1, 1, 3, 3, 3, 3, 4),
    (2, 1, 1, 1, 3, 3, 3, 4, 4),
    (2, 2, 1, 1, 3, 3, 4, 4, 4),
    (2, 2, 2, 1, 3, 4, 4, 4, 4),
    (2, 2, 2, 2, 0, 5, 5, 5, 5),
    (6, 6, 6, 6, 7, 8, 5, 5, 5),
    (6, 6, 6, 7, 7, 8, 8, 5, 5),
    (6, 6, 7, 7, 7, 8, 8, 8, 5),
    (6, 7, 7, 7, 7, 8, 8, 8, 8),
)


@dataclass
class ConfidenceThresholds:
    """Thresholds of the high-confidence pseudo-label selection."""

    tau1: float = 0.8
    tau2: float = 0.6
    tau_b: float = 0.8
    min_regions: int = 2


def _agreement_ratio(gt, i, j, label, radius):
    """Fraction of the ``(2 * radius + 1)^2`` neighbourhood equal to ``label``."""
    h, w = gt.shape
    patch = gt[max(0, i - radius):min(h, i + radius + 1),
               max(0, j - radius):min(w, j + radius + 1)]
    total = patch.size
    if total == 0:
        return 0.0
    return float(np.count_nonzero(patch == label)) / total


def intra_region_ok(gt, i, j, label, thresholds=ConfidenceThresholds()):
    """Check the label consistency of the two expansion rings."""
    if gt[i, j] != label:
        return False

    if _agreement_ratio(gt, i, j, label, radius=1) < thresholds.tau1:
        return False
    if _agreement_ratio(gt, i, j, label, radius=2) < thresholds.tau2:
        return False
    return True


def boundary_ok(gt, i, j, label, thresholds=ConfidenceThresholds()):
    """Check the label consistency of the eight directionally staggered regions.

    The centre pixel is excluded from the statistics, so this predicate only
    describes the neighbourhood; :func:`is_high_confidence` adds the check on
    the value of the pixel itself.
    """
    h, w = gt.shape
    total = [0] * 9
    match = [0] * 9

    for dx in range(-4, 5):
        x = i + dx
        if x < 0 or x >= h:
            continue
        for dy in range(-4, 5):
            y = j + dy
            if y < 0 or y >= w:
                continue
            region = _BOUNDARY_MASK[dx + 4][dy + 4]
            if region == 0:
                continue
            total[region] += 1
            if gt[x, y] == label:
                match[region] += 1

    valid_regions = 0
    for region in range(1, 9):
        if total[region] == 0:
            continue
        if match[region] / total[region] >= thresholds.tau_b:
            valid_regions += 1
            if valid_regions >= thresholds.min_regions:
                return True
    return False


def is_high_confidence(gt, i, j, label, thresholds=ConfidenceThresholds()):
    """A pixel is reliable when it is homogeneous *or* away from a boundary."""
    if gt[i, j] != label:
        return False
    return intra_region_ok(gt, i, j, label, thresholds) or boundary_ok(gt, i, j, label, thresholds)


def _window_counts(indicator, kernels):
    """Sum of ``indicator`` over every window, and the number of valid entries.

    Windows are truncated at the image border, exactly like the explicit loops
    of :func:`intra_region_ok` and :func:`boundary_ok`, which is obtained by
    zero-padding during the convolution.
    """
    ones = np.ones_like(indicator, dtype=np.float32)
    counts = [cv2.filter2D(indicator, -1, kernel, borderType=cv2.BORDER_CONSTANT)
              for kernel in kernels]
    totals = [cv2.filter2D(ones, -1, kernel, borderType=cv2.BORDER_CONSTANT)
              for kernel in kernels]
    return counts, totals


def _box_kernel(radius):
    return np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.float32)


def _region_kernels():
    """One 9x9 kernel per sub-region of the boundary mask.

    ``cv2.filter2D`` correlates the kernel with the image, so the kernel entry
    ``(u, v)`` is applied at the offset ``(u - 4, v - 4)`` -- the same indexing
    as ``_BOUNDARY_MASK``.
    """
    mask = np.asarray(_BOUNDARY_MASK)
    return [(mask == region).astype(np.float32) for region in range(1, 9)]


_BOUNDARY_KERNELS = _region_kernels()


def confidence_mask(change_map, label, thresholds=ConfidenceThresholds()):
    """Vectorised version of :func:`is_high_confidence` over a whole map.

    Returns a boolean array flagging every pixel whose value in ``change_map``
    equals ``label`` and which passes the intra-region or the boundary test.
    """
    change_map = np.asarray(change_map)
    indicator = (change_map == label).astype(np.float32)

    counts, totals = _window_counts(indicator, [_box_kernel(1), _box_kernel(2)])
    ring1 = counts[0] / np.maximum(totals[0], 1)
    ring2 = counts[1] / np.maximum(totals[1], 1)
    intra = (change_map == label) & (ring1 >= thresholds.tau1) & (ring2 >= thresholds.tau2)

    region_counts, region_totals = _window_counts(indicator, _BOUNDARY_KERNELS)
    valid = np.zeros_like(indicator, dtype=np.int32)
    for count, total in zip(region_counts, region_totals):
        ratio = count / np.maximum(total, 1)
        # Regions without any in-bounds pixel are ignored, as in boundary_ok.
        valid += ((total > 0) & (ratio >= thresholds.tau_b)).astype(np.int32)

    boundary = (change_map == label) & (valid >= thresholds.min_regions)
    return intra | boundary


def filter_high_confidence(indices, gt, label, width, thresholds=ConfidenceThresholds()):
    """Keep the subset of flat ``indices`` that passes the confidence criteria."""
    change_map = np.asarray(gt)
    if change_map.ndim == 1:
        change_map = change_map.reshape(-1, width)

    mask = confidence_mask(change_map, label, thresholds).reshape(-1)
    indices = np.asarray(indices, dtype=np.int64)
    return indices[mask[indices]]
