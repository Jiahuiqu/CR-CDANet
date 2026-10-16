"""Shared pieces of the change-detection stages."""

import os

import torch

from .models import PatchClassifier, VMUNet


def build_change_detector(config, in_chans_mod1, in_chans_mod2):
    """Build one AT-Mamba encoder/decoder and its decision-level classifier."""
    model = VMUNet(
        in_chans_mod1=in_chans_mod1,
        in_chans_mod2=in_chans_mod2,
        depths=config.get_path('cd.depths', [2, 2, 2]),
        depths_decoder=config.get_path('cd.depths_decoder', [2, 2, 2]),
        fusion_decoder=config.get_path('cd.fusion_decoder', [128, 64, 32]),
        attn_drop_rate=config.get_path('cd.attn_drop_rate', 0.2),
        drop_path_rate=config.get_path('cd.drop_path_rate', 0.2),
        num_classes=2,
    )
    patch_size = config.get_path('cd.patch_size', 17)
    classifier = PatchClassifier(spatial_size=patch_size - 1, in_channels=2)
    return model, classifier


def checkpoints_dir(config, tag):
    return config.output_dir(config.require('task.name'), 'cd', str(tag))


def save_checkpoints(config, tag, state):
    directory = checkpoints_dir(config, tag)
    os.makedirs(directory, exist_ok=True)
    torch.save(state, os.path.join(directory, 'checkpoint.pt'))
    return directory


def load_checkpoints(config, tag, device):
    path = os.path.join(checkpoints_dir(config, tag), 'checkpoint.pt')
    if not os.path.exists(path):
        raise FileNotFoundError(f"No checkpoint of round '{tag}' found at {path}")
    return torch.load(path, map_location=device)


def loss_weights_for_epoch(schedule, epoch):
    """``(main_weight, alignment_weight)`` of the current epoch.

    ``schedule`` is a list of ``{until_epoch, main_weight, alignment_weight}``
    entries evaluated in order; ``until_epoch: null`` closes the list.
    """
    for entry in schedule:
        limit = entry.get('until_epoch')
        if limit is None or epoch < limit:
            return entry['main_weight'], entry['alignment_weight']
    last = schedule[-1]
    return last['main_weight'], last['alignment_weight']
