"""Multi-scale joint domain adaptation loss.

The features of the source and target domains are aligned at every scale of the
encoder.  Two statistics are combined, with a level-dependent weight:

    L_DA = sum_l [ alpha_l * MMD(F_s^l, F_t^l) + (1 - alpha_l) * CORAL(F_s^l, F_t^l) ]

``alpha_l`` grows with the depth of the level: the shallow features are mostly
textural, where matching the covariance (CORAL) is more informative, while the
deep features carry semantic content, where matching the distribution (MMD) is
more effective.  The default ``(0.3, 0.5, 0.7)`` follows the paper's setting.
"""

import torch.nn as nn

from .coral import CoralLoss
from .mmd import MMDLoss


class MultiScaleAlignmentLoss(nn.Module):
    """Weighted combination of MMD and CORAL over the encoder scales.

    Args:
        alphas: MMD weight of each scale, ordered from the shallowest to the
            deepest.  ``1 - alpha`` is used for CORAL.
    """

    def __init__(self, alphas=(0.3, 0.5, 0.7)):
        super().__init__()
        self.alphas = tuple(alphas)
        self.coral_loss = CoralLoss()
        self.mmd_loss = MMDLoss()

    def calculate_losses(self, features_source, features_target):
        if len(features_source) != len(features_target):
            raise ValueError("Source and target feature lists must have the same length")

        total_coral_loss = 0.0
        total_mmd_loss = 0.0
        total_weight = 0.0

        for level, (source, target) in enumerate(zip(features_source, features_target)):
            alpha = self.alphas[level] if level < len(self.alphas) else self.alphas[-1]

            source_flat = source.reshape(source.size(0), -1)
            target_flat = target.reshape(target.size(0), -1)

            total_coral_loss = total_coral_loss + self.coral_loss(source_flat, target_flat) * (1.0 - alpha)
            total_mmd_loss = total_mmd_loss + self.mmd_loss(source=source_flat, target=target_flat) * alpha
            total_weight += 1.0

        return total_coral_loss / total_weight, total_mmd_loss / total_weight

    def forward(self, features_source, features_target):
        total_coral_loss, total_mmd_loss = self.calculate_losses(features_source, features_target)
        return total_coral_loss, total_mmd_loss, total_coral_loss + total_mmd_loss
