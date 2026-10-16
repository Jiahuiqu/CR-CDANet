"""Stage 3b: dense prediction on the target domain and evaluation.

Every labelled pixel of the target domain is classified from its own
``patch_size x patch_size`` neighbourhood; the change map is written as a .mat
file so that it can supervise the next refinement round.

    python evaluate.py --config configs/shuguang2islandtown.yaml \\
        --tag 0 --target-tag 0 --predict-tag round_0
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config import load_config
from src.data import ChangePatchDataset, load_cross_modal, load_raw_dataset
from src.engine import build_change_detector, load_checkpoints
from src.metrics import calculate_metrics, format_metrics
from src.utils import save_mat, set_seed


def predict(model, classifier, loader, device):
    """Run the detector on every patch and return the predicted change map."""
    model.eval()
    classifier.eval()

    predictions = []
    positions = []

    start = time.time()
    with torch.no_grad():
        for pre_rgb, post_rgb, pre_sar, post_sar, _label, position in loader:
            logits, *_ = model([pre_rgb.to(device).float(), post_rgb.to(device).float()],
                               [pre_sar.to(device).float(), post_sar.to(device).float()])
            predicted = classifier(logits).argmax(dim=1)

            predictions.append(predicted.cpu().numpy())
            positions.append(position.numpy())

    return np.concatenate(predictions), np.concatenate(positions), time.time() - start


def main():
    parser = argparse.ArgumentParser(description="Evaluate a change detection checkpoint on the target domain")
    parser.add_argument('--config', required=True)
    parser.add_argument('--tag', required=True, help='Tag of the checkpoint to evaluate')
    parser.add_argument('--target-tag', default=None,
                        help='Tag of the cross-modal images of the target (defaults to --tag)')
    parser.add_argument('--predict-tag', default=None,
                        help='Folder the change map is written to (defaults to --tag)')
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device(config.get_path('device', 'cuda:0'))
    set_seed(config.get_path('seed', 42))

    task = config.require('task.name')
    target_name = config.require('task.target')
    target_tag = args.target_tag if args.target_tag is not None else args.tag
    predict_tag = args.predict_tag if args.predict_tag is not None else args.tag

    target_raw = load_raw_dataset(config, target_name)
    pair = load_cross_modal(config, target_raw, task, target_tag)

    labelled = np.flatnonzero(target_raw.labelled)
    dataset = ChangePatchDataset(
        [pair.rgb_synthesized, pair.rgb_original, pair.sar_synthesized, pair.sar_original],
        target_raw.change_map, labelled, patch_size=config.get_path('data.patch_size', 17))
    loader = DataLoader(dataset,
                        batch_size=config.get_path('cd.eval_batch_size', 256),
                        shuffle=False, num_workers=0, drop_last=False)

    model, classifier = build_change_detector(config, pair.rgb_synthesized.shape[2], pair.sar_synthesized.shape[2])
    checkpoint = load_checkpoints(config, args.tag, device)
    model.load_state_dict(checkpoint['model'])
    classifier.load_state_dict(checkpoint['classifier'])
    model, classifier = model.to(device), classifier.to(device)

    predictions, positions, elapsed = predict(model, classifier, loader, device)

    change_map = np.zeros(target_raw.shape, dtype=np.float32)
    change_map[positions[:, 0], positions[:, 1]] = predictions.astype(np.float32)

    directory = config.output_dir(task, 'predictions', target_name, predict_tag)
    save_mat(os.path.join(directory, 'predict_mat.mat'), 'pred', change_map)

    labels = target_raw.change_map[positions[:, 0], positions[:, 1]]
    oa, kappa, f1, pr, re = calculate_metrics(predictions.astype(np.float64), labels.astype(np.float64))

    print(f"[evaluate:{predict_tag}] dataset={target_name} pixels={len(predictions)} time={elapsed:.1f}s")
    print(f"[evaluate:{predict_tag}] {format_metrics(oa, kappa, f1, pr, re)}")
    print(f"[evaluate:{predict_tag}] change map saved to {directory}")

    return oa, kappa, f1


if __name__ == '__main__':
    main()
