"""Decision-level classifier.

The change-detection backbone outputs a ``(B, num_classes, P-1, P-1)`` logit map
for a ``P x P`` input patch.  The classifier flattens the spatial dimensions of
that map and applies an MLP to each class channel, yielding one logit per class
and per patch.  This is the decision-level head used both for the source and the
target domain (they share this definition and only differ in their parameters).
"""

import torch.nn as nn


class PatchClassifier(nn.Module):
    """MLP over the flattened logit map of a patch.

    Args:
        spatial_size: Side length of the logit map (``patch_size - 1``).
        in_channels: Number of class channels of the logit map.
    """

    def __init__(self, spatial_size=16, in_channels=2, hidden_dims=(512, 256, 128, 64, 32, 16, 4)):
        super().__init__()
        self.flatten = nn.Flatten(-2)
        self.relu = nn.ReLU()

        dims = (spatial_size * spatial_size,) + tuple(hidden_dims)
        layers = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
        self.layers = nn.ModuleList(layers)
        self.out = nn.Linear(dims[-1], 1)

    def forward(self, x):
        x = self.relu(self.layers[0](self.flatten(x)))
        for layer in self.layers[1:]:
            x = self.relu(layer(x))
        return self.out(x).squeeze(-1)
