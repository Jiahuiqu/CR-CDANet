"""Stage 2: supervised training of the change detector on the source domain.

The resulting model initialises the source branch of the first refinement round
and provides the preliminary change map of the target domain, which weakly
guides the patch selection of DADiff.

    python train_source.py --config configs/shuguang2islandtown.yaml
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config import load_config
from src.data import ChangePatchDataset, load_cross_modal, load_raw_dataset, sample_source_points
from src.engine import build_change_detector, save_checkpoints
from src.utils import set_seed


def train_epoch(model, classifier, loader, optimizer, device, epoch, epochs):
    model.train()
    classifier.train()

    total_loss = 0.0
    for pre_rgb, post_rgb, pre_sar, post_sar, label, _ in tqdm(loader, desc=f'epoch {epoch + 1}/{epochs}'):
        pre_rgb = pre_rgb.to(device).float()
        post_rgb = post_rgb.to(device).float()
        pre_sar = pre_sar.to(device).float()
        post_sar = post_sar.to(device).float()
        label = label.to(device).long()

        optimizer.zero_grad()

        logits, *_ = model([pre_rgb, post_rgb], [pre_sar, post_sar])
        loss = F.cross_entropy(classifier(logits), label)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


def main():
    parser = argparse.ArgumentParser(description="Train the change detector on the source domain")
    parser.add_argument('--config', required=True)
    parser.add_argument('--dataset', default=None, help='Defaults to task.source')
    parser.add_argument('--tag', default='source', help='Name of the saved checkpoint')
    args = parser.parse_args()

    config = load_config(args.config)
    dataset_name = args.dataset or config.require('task.source')
    device = torch.device(config.get_path('device', 'cuda:0'))
    set_seed(config.get_path('seed', 42))

    raw = load_raw_dataset(config, dataset_name)
    pair = load_cross_modal(config, raw, config.require('task.name'), 'source')

    ratio = config.get_path('cd.source_label_ratio', 1.0)
    points = sample_source_points(raw.change_map, raw.labelled, ratio, config.get_path('seed', 42))
    print(f"[source] dataset={dataset_name} label ratio={ratio} -> {len(points)} training pixels")

    dataset = ChangePatchDataset(
        [pair.rgb_synthesized, pair.rgb_original, pair.sar_synthesized, pair.sar_original],
        raw.change_map, points, patch_size=config.get_path('data.patch_size', 17))
    loader = DataLoader(dataset,
                        batch_size=config.get_path('cd.batch_size', 128),
                        shuffle=True,
                        num_workers=config.get_path('cd.num_workers', 4),
                        pin_memory=True)

    model, classifier = build_change_detector(config, pair.rgb_synthesized.shape[2], pair.sar_synthesized.shape[2])
    model = model.to(device)
    classifier = classifier.to(device)

    epochs = config.get_path('cd.epochs', 100)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(classifier.parameters()),
        lr=config.get_path('cd.lr', 1e-4),
        weight_decay=config.get_path('cd.weight_decay', 5e-4))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    for epoch in range(epochs):
        average_loss = train_epoch(model, classifier, loader, optimizer, device, epoch, epochs)
        scheduler.step()
        print(f"[source] epoch {epoch + 1}/{epochs} loss {average_loss:.6f}")

    directory = save_checkpoints(config, args.tag, {
        'model': model.state_dict(),
        'classifier': classifier.state_dict(),
    })
    print(f"[source] checkpoint saved to {directory}")


if __name__ == '__main__':
    main()
