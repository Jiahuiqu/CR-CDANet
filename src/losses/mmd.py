import torch
import torch.nn as nn


class MMDLoss(nn.Module):
    """Maximum mean discrepancy between the source and target features.

    The squared MMD is estimated with a multi-bandwidth RBF kernel:

        MMD^2 = E[k(x, x')] + E[k(y, y')] - 2 * E[k(x, y)]

    Args:
        kernel_mul: Ratio between consecutive kernel bandwidths.
        kernel_num: Number of RBF kernels.
        fix_sigma: Fixed bandwidth; if ``None`` the median heuristic is used.
    """

    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None, **kwargs):
        super().__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul, kernel_num, fix_sigma):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        l2_distance = ((total0 - total1) ** 2).sum(2)

        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(l2_distance.data) / (n_samples ** 2 - n_samples)

        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-l2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_x, f_of_y):
        delta = f_of_x.float().mean(0) - f_of_y.float().mean(0)
        return delta.dot(delta.T)

    def forward(self, source, target):
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)

        if self.kernel_type != 'rbf':
            raise NotImplementedError(f"Unknown kernel type: {self.kernel_type}")

        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(
            source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        xx = torch.mean(kernels[:batch_size, :batch_size])
        yy = torch.mean(kernels[batch_size:, batch_size:])
        xy = torch.mean(kernels[:batch_size, batch_size:])
        yx = torch.mean(kernels[batch_size:, :batch_size])
        return torch.mean(xx + yy - xy - yx)
