"""Torch datasets used by the two training stages."""

import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ..refinement.pseudo_label import ConfidenceThresholds, confidence_mask


def _confident_unchanged_mask(change_map, labelled, thresholds):
    """Boolean mask of the pixels that are reliable *and* unchanged."""
    mask = confidence_mask(change_map, 0, thresholds)
    if labelled is not None:
        mask = mask & labelled
    return mask.astype(np.uint8)


class DiffusionPatchDataset(Dataset):
    """Crops used to train DADiff on unchanged regions.

    Only crops whose pixels are all high-confidence unchanged are kept, which
    follows the paper's rule of excluding patches that contain changed,
    boundary or low-confidence pixels.

    Args:
        condition: image of the modality used as the condition, ``(H, W, C)``.
        target: image of the modality to be generated, ``(H, W, C')``.
        guidance_map: change map (0 = unchanged, 1 = changed) used to reject
            crops.  For the first stage it is the ground truth of the dataset,
            afterwards the prediction of the previous refinement round.
        labelled: mask of the reliable pixels of ``guidance_map`` (optional).
        crop_size: side length of the crops.
        num_samples: number of crops of the training split.
    """

    def __init__(self, condition, target, guidance_map, labelled=None, crop_size=16,
                 num_samples=20000, mode='train', train_split=0.9999, seed=42,
                 thresholds=ConfidenceThresholds()):
        assert mode in ('train', 'val', 'test')

        self.crop_size = crop_size
        self.num_samples = num_samples
        self.mode = mode
        self.train_split = train_split

        self.condition = np.pad(
            np.asarray(condition, dtype=np.float32),
            ((0, (crop_size - condition.shape[0] % crop_size) % crop_size),
             (0, (crop_size - condition.shape[1] % crop_size) % crop_size), (0, 0)),
            mode='reflect')
        self.target = np.pad(
            np.asarray(target, dtype=np.float32),
            ((0, (crop_size - target.shape[0] % crop_size) % crop_size),
             (0, (crop_size - target.shape[1] % crop_size) % crop_size), (0, 0)),
            mode='reflect')

        self.height, self.width = guidance_map.shape

        if mode == 'test':
            self.positions = [(i, j)
                              for i in range(0, self.height, crop_size)
                              for j in range(0, self.width, crop_size)]
            return

        mask = _confident_unchanged_mask(guidance_map, labelled, thresholds)
        integral = cv2.integral(mask.astype(np.float32))
        area = float(crop_size * crop_size)

        candidates = []
        for i in range(0, self.height - crop_size + 1):
            for j in range(0, self.width - crop_size + 1):
                total = (integral[i + crop_size, j + crop_size] - integral[i, j + crop_size]
                         - integral[i + crop_size, j] + integral[i, j])
                if total == area:
                    candidates.append((i, j))

        if not candidates:
            raise RuntimeError(
                "No crop without changed or low-confidence pixels was found. "
                "Check the change map and the confidence thresholds.")

        rng = random.Random(seed)
        if len(candidates) >= num_samples:
            positions = rng.sample(candidates, num_samples)
        else:
            positions = [rng.choice(candidates) for _ in range(num_samples)]

        split = int(num_samples * train_split)
        self.positions = positions[:split] if mode == 'train' else positions[split:]
        self.original_height, self.original_width = guidance_map.shape

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, index):
        i, j = self.positions[index]
        size = self.crop_size

        condition = self.condition[i:i + size, j:j + size].transpose(2, 0, 1)
        target = self.target[i:i + size, j:j + size].transpose(2, 0, 1)

        return (torch.from_numpy(np.ascontiguousarray(condition)),
                torch.from_numpy(np.ascontiguousarray(target)))

    def reconstruct(self, patches):
        """Tile a list of generated crops back into the padded image."""
        channels = patches[0].shape[0]
        image = torch.zeros((channels, self.condition.shape[0], self.condition.shape[1]))

        index = 0
        for i in range(0, self.condition.shape[0], self.crop_size):
            for j in range(0, self.condition.shape[1], self.crop_size):
                image[:, i:i + self.crop_size, j:j + self.crop_size] = patches[index]
                index += 1

        return image[:, :self.height, :self.width]


class ChangePatchDataset(Dataset):
    """``patch_size x patch_size`` patches centred on the selected pixels.

    The label of a patch is the class of its centre pixel, which is what the
    classifier head predicts.
    """

    def __init__(self, images, change_map, points, patch_size=17):
        if len(images) != 4:
            raise ValueError("Expected the four images of the cross-modal pair")

        self.patch_size = patch_size
        self.padding = patch_size // 2
        self.change_map = change_map
        self.height, self.width = change_map.shape

        self.padded = []
        for image in images:
            image = np.asarray(image, dtype=np.float32)
            if image.ndim == 2:
                image = image[:, :, None]
            self.padded.append(cv2.copyMakeBorder(
                image, self.padding, self.padding, self.padding, self.padding, cv2.BORDER_REPLICATE))

        self.points = np.asarray(points, dtype=np.int64)

    def __len__(self):
        return len(self.points)

    def __getitem__(self, index):
        flat = int(self.points[index])
        i, j = divmod(flat, self.width)
        new_i, new_j = i + self.padding, j + self.padding

        patches = []
        for image in self.padded:
            patch = image[new_i - self.padding:new_i + self.padding + 1,
                          new_j - self.padding:new_j + self.padding + 1, :].transpose(2, 0, 1)
            patches.append(torch.from_numpy(np.ascontiguousarray(patch)))

        label = torch.tensor(self.change_map[i, j], dtype=torch.float32)
        position = torch.tensor((i, j), dtype=torch.long)
        return patches[0], patches[1], patches[2], patches[3], label, position
