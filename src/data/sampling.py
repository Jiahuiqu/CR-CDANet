"""Selection of the labelled pixels that are used to train the change detector.

Two sources of supervision are combined across the refinement rounds:

* the ground-truth labels of the target domain, of which only a small fraction
  (1% by default, i.e. 1% of each class) is revealed;
* the high-confidence pseudo labels of the change map predicted by the previous
  round, which become available incrementally.

Round ``r`` therefore trains on ``ratios[r]`` of each class, made of the 1%
ground-truth sample plus ``ratios[r] - 0.01`` taken from the confident pseudo
labels.
"""

import random

import numpy as np

from ..refinement.pseudo_label import ConfidenceThresholds, filter_high_confidence

INITIAL_LABEL_RATIO = 0.01


def class_indices(change_map, labelled):
    """Flat indices of the labelled changed and unchanged pixels."""
    changed = np.flatnonzero((change_map == 1) & labelled)
    unchanged = np.flatnonzero((change_map == 0) & labelled)
    return changed, unchanged


def _sample(indices, count, rng, allow_replacement=True):
    """Deterministic sampling of ``count`` entries out of ``indices``."""
    indices = list(indices)
    if count <= 0 or not indices:
        return []
    if len(indices) >= count:
        return rng.sample(indices, count)
    if not allow_replacement:
        return list(indices)
    return [rng.choice(indices) for _ in range(count)]


def sample_source_points(change_map, labelled, ratio, seed):
    """Fully or partially supervised sampling on a labelled domain."""
    rng = random.Random(seed)
    changed, unchanged = class_indices(change_map, labelled)
    changed_points = _sample(changed, int(len(changed) * ratio), rng)
    unchanged_points = _sample(unchanged, int(len(unchanged) * ratio), rng)
    return np.asarray(changed_points + unchanged_points, dtype=np.int64)


def sample_round_points(change_map, change_map_real, labelled, ratios, round_index,
                        thresholds=ConfidenceThresholds(), seed=42, width=None):
    """Training points of refinement round ``round_index``.

    Args:
        change_map: change map used for supervision (the ground truth for round 0,
            the prediction of the previous round afterwards).
        change_map_real: ground-truth change map of the target domain; it provides
            the revealed labels.
        labelled: mask of the evaluated pixels.
        ratios: target label ratio of every round, e.g. ``[0.01, 0.05, ..., 0.20]``.
        round_index: index into ``ratios``.
        width: image width, needed to test the confidence of a flat index.
        seed: seed of the sampling.
    """
    ratio = ratios[round_index]
    rng = random.Random(seed)

    if width is None:
        width = change_map.shape[1]

    if round_index == 0:
        changed, unchanged = class_indices(change_map, labelled)
        changed_points = _sample(changed, int(len(changed) * ratio), rng)
        unchanged_points = _sample(unchanged, int(len(unchanged) * ratio), rng)
        return np.asarray(changed_points + unchanged_points, dtype=np.int64)

    # Revealed ground truth: the 1% sample that the paper assumes to be available.
    real_changed, real_unchanged = class_indices(change_map_real, labelled)
    base_changed = _sample(real_changed, int(len(real_changed) * INITIAL_LABEL_RATIO), rng)
    base_unchanged = _sample(real_unchanged, int(len(real_unchanged) * INITIAL_LABEL_RATIO), rng)

    # Incremental part: additional high-confidence pseudo labels.  The target of
    # each class is computed once, so that the rounding of the initial 1% cannot
    # make the total drift away from ``ratio``.
    count_changed = max(int(len(real_changed) * ratio) - len(base_changed), 0)
    count_unchanged = max(int(len(real_unchanged) * ratio) - len(base_unchanged), 0)

    confident_changed = filter_high_confidence(
        np.flatnonzero(change_map == 1), change_map.astype(np.int64), 1, width, thresholds)
    confident_unchanged = filter_high_confidence(
        np.flatnonzero(change_map == 0), change_map.astype(np.int64), 0, width, thresholds)

    remaining_changed = list(set(confident_changed) - set(base_changed))
    remaining_unchanged = list(set(confident_unchanged) - set(base_unchanged))

    changed_points = _sample(remaining_changed, count_changed, rng)
    unchanged_points = _sample(remaining_unchanged, count_unchanged, rng)
    changed_points.extend(base_changed)
    unchanged_points.extend(base_unchanged)

    return np.asarray(changed_points + unchanged_points, dtype=np.int64)
