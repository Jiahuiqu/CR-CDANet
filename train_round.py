"""Stage 3a: one refinement round of the change detector on the target domain.

The round implements the domain adaptation and co-training based change
detection (DCCD) of the paper:

* the target and source encoders are trained jointly with a supervised loss on
  their own labelled pixels;
* the multi-scale features of the target encoder are aligned with those of the
  source encoder using the dynamic MMD / CORAL fusion;
* the two classifiers are swapped on alternating epochs, so that each domain is
  also classified by the decision boundary of the other one.

    python train_round.py --config configs/shuguang2islandtown.yaml --round 0
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
from src.data import (ChangePatchDataset, load_cross_modal, load_raw_dataset,
                      sample_round_points, sample_source_points)
from src.engine import build_change_detector, load_checkpoints, loss_weights_for_epoch, save_checkpoints
from src.losses import MultiScaleAlignmentLoss
from src.refinement.pseudo_label import ConfidenceThresholds
from src.utils import CyclicLoader, load_mat, set_seed


def align_features(alignment, target_features, source_features):
    """Align the four target feature maps with their source counterparts.

    All four target feature maps -- the two branches of both modalities -- are
    matched against the source features of the ``x`` branch, which is the pairing
    used by the reference implementation (Sec. III-C.1 of the paper).
    """
    enc_m1_ctx, enc_m1_x, enc_m2_ctx, enc_m2_x = target_features
    _, src_m1_x, _, src_m2_x = source_features

    pairs = ((enc_m1_ctx, src_m1_x), (enc_m1_x, src_m1_x),
             (enc_m2_ctx, src_m2_x), (enc_m2_x, src_m2_x))

    total_coral = 0.0
    total_mmd = 0.0
    for target_feature, source_feature in pairs:
        coral, mmd, _ = alignment(source_feature, target_feature)
        total_coral = total_coral + coral
        total_mmd = total_mmd + mmd
    return total_coral, total_mmd, total_coral + total_mmd


def main():
    parser = argparse.ArgumentParser(description="Train one refinement round on the target domain")
    parser.add_argument('--config', required=True)
    parser.add_argument('--round', type=int, required=True, help='Refinement round index')
    parser.add_argument('--source-tag', default='source', help='Checkpoint tag of the source domain')
    parser.add_argument('--target-tag', default=None,
                        help='Tag of the cross-modal images used for the target domain '
                             '(defaults to the round index)')
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device(config.get_path('device', 'cuda:0'))
    set_seed(config.get_path('seed', 42))

    task = config.require('task.name')
    source_name = config.require('task.source')
    target_name = config.require('task.target')

    ratios = config.get_path('cd.target_label_ratios', [0.01, 0.05, 0.10, 0.15, 0.20])
    if args.round >= len(ratios):
        raise ValueError(f"Round {args.round} is beyond the configured label ratios {ratios}")

    thresholds = ConfidenceThresholds(
        tau1=config.get_path('refinement.tau1', 0.8),
        tau2=config.get_path('refinement.tau2', 0.6),
        tau_b=config.get_path('refinement.tau_b', 0.8),
        min_regions=config.get_path('refinement.min_regions', 2))

    patch_size = config.get_path('data.patch_size', 17)

    # ---- data -----------------------------------------------------------------
    source_raw = load_raw_dataset(config, source_name)
    source_pair = load_cross_modal(config, source_raw, task, args.source_tag)
    source_points = sample_source_points(
        source_raw.change_map, source_raw.labelled,
        config.get_path('cd.source_label_ratio', 1.0), config.get_path('seed', 42))
    source_dataset = ChangePatchDataset(
        [source_pair.rgb_synthesized, source_pair.rgb_original,
         source_pair.sar_synthesized, source_pair.sar_original],
        source_raw.change_map, source_points, patch_size=patch_size)

    target_raw = load_raw_dataset(config, target_name)
    target_tag = str(args.target_tag) if args.target_tag is not None else str(args.round)
    target_pair = load_cross_modal(config, target_raw, task, target_tag)

    if args.round == 0:
        target_map = target_raw.change_map
    else:
        prediction = os.path.join(config.output_dir(task, 'predictions', target_name,
                                                    f'round_{args.round - 1}'), 'predict_mat.mat')
        target_map = load_mat(prediction, 'pred')
    target_points = sample_round_points(
        target_map, target_raw.change_map, target_raw.labelled, ratios, args.round,
        thresholds=thresholds, seed=config.get_path('seed', 42), width=target_raw.shape[1])
    print(f"[round {args.round}] target label ratio {ratios[args.round]} -> {len(target_points)} training pixels")

    target_dataset = ChangePatchDataset(
        [target_pair.rgb_synthesized, target_pair.rgb_original,
         target_pair.sar_synthesized, target_pair.sar_original],
        target_map, target_points, patch_size=patch_size)

    batch_size = config.get_path('cd.batch_size', 128)
    num_workers = config.get_path('cd.num_workers', 4)
    target_loader = DataLoader(target_dataset, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=True)
    source_loader = CyclicLoader(DataLoader(source_dataset, batch_size=batch_size, shuffle=True,
                                            num_workers=num_workers, pin_memory=True))

    # ---- model ----------------------------------------------------------------
    in_chans_rgb = target_pair.rgb_synthesized.shape[2]
    in_chans_sar = target_pair.sar_synthesized.shape[2]

    model, classifier = build_change_detector(config, in_chans_rgb, in_chans_sar)
    model_source, classifier_source = build_change_detector(config, in_chans_rgb, in_chans_sar)
    model, classifier = model.to(device), classifier.to(device)
    model_source, classifier_source = model_source.to(device), classifier_source.to(device)

    previous_tag = args.source_tag if args.round == 0 else str(args.round - 1)
    previous = load_checkpoints(config, previous_tag, device)
    # A source-domain checkpoint holds a single branch, which initialises the
    # source branch of the first round; later rounds provide both branches.
    previous_model_source = previous.get('model_source', previous['model'])
    previous_classifier_source = previous.get('classifier_source', previous['classifier'])
    if args.round == 0:
        # The target branch starts from scratch, the source branch from the
        # model trained on the source domain.
        print(f"[round {args.round}] source branch loaded from '{previous_tag}'")
    else:
        model.load_state_dict(previous['model'])
        classifier.load_state_dict(previous['classifier'])
        print(f"[round {args.round}] target branch loaded from '{previous_tag}'")
    model_source.load_state_dict(previous_model_source)
    classifier_source.load_state_dict(previous_classifier_source)

    network = [model, classifier, model_source, classifier_source]
    optimizers = [torch.optim.AdamW(module.parameters(),
                                    lr=config.get_path('cd.lr', 1e-4),
                                    weight_decay=config.get_path('cd.weight_decay', 5e-4))
                  for module in network]
    schedulers = [torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
                  for optimizer in optimizers]

    alignment = MultiScaleAlignmentLoss(
        alphas=config.get_path('cd.alphas', [0.3, 0.5, 0.7]))

    schedule = config.get_path('cd.loss_schedule')
    epochs = config.get_path('cd.epochs', 100)

    # ---- training -------------------------------------------------------------
    for epoch in range(epochs):
        main_weight, alignment_weight = loss_weights_for_epoch(schedule, epoch)

        # The classifiers are swapped on alternating epochs (Eq. 21-22).
        target_head = classifier if epoch % 2 == 0 else classifier_source
        source_head = classifier_source if epoch % 2 == 0 else classifier

        model.train()
        model_source.train()
        classifier.train()
        classifier_source.train()

        total_target_loss = 0.0
        total_source_loss = 0.0
        total_alignment = 0.0

        for pre_rgb, post_rgb, pre_sar, post_sar, target_label, _ in tqdm(
                target_loader, desc=f'round {args.round} epoch {epoch + 1}/{epochs}'):
            src_pre_rgb, src_post_rgb, src_pre_sar, src_post_sar, source_label, _ = source_loader.next()

            pre_rgb, post_rgb = pre_rgb.to(device).float(), post_rgb.to(device).float()
            pre_sar, post_sar = pre_sar.to(device).float(), post_sar.to(device).float()
            target_label = target_label.to(device).long()

            src_pre_rgb, src_post_rgb = src_pre_rgb.to(device).float(), src_post_rgb.to(device).float()
            src_pre_sar, src_post_sar = src_pre_sar.to(device).float(), src_post_sar.to(device).float()
            source_label = source_label.to(device).long()

            for optimizer in optimizers:
                optimizer.zero_grad()

            target_logits, *target_features = model(
                [pre_rgb, post_rgb], [pre_sar, post_sar])
            source_logits, *source_features = model_source(
                [src_pre_rgb, src_post_rgb], [src_pre_sar, src_post_sar])

            loss_target = F.cross_entropy(target_head(target_logits), target_label)
            loss_source = F.cross_entropy(source_head(source_logits), source_label)

            _, _, alignment_loss = align_features(alignment, target_features, source_features)

            final_loss = main_weight * (loss_target + loss_source) + alignment_weight * alignment_loss
            final_loss.backward()

            for optimizer in optimizers:
                optimizer.step()

            total_target_loss += loss_target.item()
            total_source_loss += loss_source.item()
            total_alignment += float(alignment_loss)

        batches = max(len(target_loader), 1)
        print(f"[round {args.round}] epoch {epoch + 1}/{epochs} "
              f"target loss {total_target_loss / batches:.6f} "
              f"source loss {total_source_loss / batches:.6f} "
              f"alignment loss {total_alignment / batches:.6f} "
              f"(main weight {main_weight}, alignment weight {alignment_weight})")

        for scheduler in schedulers:
            scheduler.step()

    directory = save_checkpoints(config, str(args.round), {
        'model': model.state_dict(),
        'classifier': classifier.state_dict(),
        'model_source': model_source.state_dict(),
        'classifier_source': classifier_source.state_dict(),
    })
    print(f"[round {args.round}] checkpoint saved to {directory}")


if __name__ == '__main__':
    main()
