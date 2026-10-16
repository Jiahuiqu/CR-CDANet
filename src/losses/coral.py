import torch
import torch.nn as nn


class CoralLoss(nn.Module):
    """CORAL: squared Frobenius distance between the two covariance matrices.

        L = || Cov_s - Cov_t ||_F^2 / (4 * d^2)

    where ``d`` is the feature dimension.  Aligning second-order statistics
    complements MMD, which only matches first-order statistics.
    """

    def __init__(self):
        super().__init__()

    def forward(self, source, target):
        d = source.size(1)

        source_mean = source.mean(0)
        target_mean = target.mean(0)

        source_centered = source - source_mean
        target_centered = target - target_mean

        source_corr = torch.matmul(source_centered.t(), source_centered) / (source.size(0) - 1)
        target_corr = torch.matmul(target_centered.t(), target_centered) / (target.size(0) - 1)

        return torch.sum(torch.pow(source_corr - target_corr, 2)) / (4 * d * d)
