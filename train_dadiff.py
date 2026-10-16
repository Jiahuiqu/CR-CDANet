"""Stage 1: training of DADiff, the cross-modal diffusion translation module.

Two modes are provided:

``--stage pretrain``
    Train the diffusion model on the *source* domain, where it learns to
    translate between the two modalities.  Its output provides the cross-modal
    images of the source branch of every refinement round, and it is the initial
    state of the target-domain model.

``--stage adapt``
    Fine-tune the model on the *target* domain.  The diffusion loss is combined
    with the multi-scale joint domain adaptation loss (MMD / CORAL) computed
    between the conditional-encoder features of the target model and those of the
    frozen source model.  The crops are taken from the regions that the previous
    refinement round considers unchanged.

Both directions are trained (``rgb`` from the SAR image and ``sar`` from the RGB
image) and the translated images are written as .mat files.

    python train_dadiff.py --config configs/shuguang2islandtown.yaml \\
        --stage pretrain --dataset shuguang
    python train_dadiff.py --config configs/shuguang2islandtown.yaml \\
        --stage adapt --dataset islandtown --round 0 --guidance-map <predict_mat.mat>
"""

import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config import load_config
from src.dadiff import DADiff
from src.data import DiffusionPatchDataset, load_raw_dataset
from src.refinement.pseudo_label import ConfidenceThresholds
from src.utils import CyclicLoader, load_mat, save_mat, set_seed

# The direction names state what is generated; the tuple is (condition, target).
DIRECTIONS = {
    'rgb': ('sar', 'rgb'),
    'sar': ('rgb', 'sar'),
}


def models_dir(config, dataset_name, tag):
    return config.output_dir(config.require('task.name'), 'dadiff_models', dataset_name, str(tag))


def images_dir(config, dataset_name, tag):
    return config.output_dir(config.require('task.name'), 'dadiff', dataset_name, str(tag))


def build_patch_dataset(config, raw, direction, guidance_map, labelled, mode):
    condition_modality, target_modality = DIRECTIONS[direction]
    thresholds = ConfidenceThresholds(
        tau1=config.get_path('refinement.tau1', 0.8),
        tau2=config.get_path('refinement.tau2', 0.6),
        tau_b=config.get_path('refinement.tau_b', 0.8),
        min_regions=config.get_path('refinement.min_regions', 2))

    return DiffusionPatchDataset(
        getattr(raw, condition_modality), getattr(raw, target_modality),
        guidance_map, labelled=labelled,
        crop_size=config.get_path('data.diffusion_patch_size', 16),
        num_samples=config.get_path('dadiff.num_samples', 20000),
        mode=mode,
        train_split=config.get_path('dadiff.train_split', 0.9999),
        seed=config.get_path('seed', 42),
        thresholds=thresholds)


def build_dadiff(config, raw, direction, device, use_alignment):
    condition_modality, target_modality = DIRECTIONS[direction]
    return DADiff(direction=direction,
                  in_chans=getattr(raw, target_modality).shape[2],
                  device=device,
                  timesteps=config.get_path('dadiff.timesteps', 2000),
                  sample_steps=config.get_path('dadiff.sample_steps', 100),
                  loss_type=config.get_path('dadiff.loss_type', 'l2'),
                  lr=config.get_path('dadiff.lr', 1e-4),
                  alphas=config.get_path('dadiff.alphas', [0.3, 0.5, 0.7]),
                  use_alignment=use_alignment)


def train_direction(config, raw, direction, tag, guidance_map, labelled, epochs, device,
                    source_raw=None):
    """Train one translation direction, save the model and write its output."""
    dataset = build_patch_dataset(config, raw, direction, guidance_map, labelled, 'train')
    loader = DataLoader(dataset,
                        batch_size=config.get_path('dadiff.batch_size', 256),
                        shuffle=True,
                        num_workers=config.get_path('dadiff.num_workers', 4),
                        pin_memory=True)

    adapting = source_raw is not None
    dadiff = build_dadiff(config, raw, direction, device, use_alignment=adapting)

    # The frozen reference of the adaptation stage is the source-domain model.
    reference_loader = None
    if adapting:
        init_path = os.path.join(models_dir(config, source_raw.name, 'source'), f'{direction}.pt')
        if not os.path.exists(init_path):
            raise FileNotFoundError(
                f"Source-domain model not found at {init_path}; run the pre-training stage first")
        dadiff.netG.load_state_dict(torch.load(init_path, map_location=device)['netG'])
        dadiff.source_netG.load_state_dict(dadiff.netG.state_dict())
        for parameter in dadiff.source_netG.parameters():
            parameter.requires_grad = False
        dadiff.source_netG.model.eval()
        print(f"[{direction}] initialised from {init_path}")

        reference_dataset = build_patch_dataset(
            config, source_raw, direction, source_raw.change_map, source_raw.labelled, 'train')
        reference_loader = CyclicLoader(DataLoader(
            reference_dataset,
            batch_size=config.get_path('dadiff.batch_size', 256),
            shuffle=True, num_workers=2, pin_memory=True))

    for epoch in range(epochs):
        dadiff.netG.train()
        total_loss = 0.0
        total_alignment = 0.0

        for condition_batch, target_batch in tqdm(loader, desc=f'{direction} epoch {epoch + 1}/{epochs}'):
            condition_batch = condition_batch.to(device).float()
            target_batch = target_batch.to(device).float()

            if reference_loader is None:
                loss, _, alignment_loss, _, _ = dadiff.optimize_parameters(
                    condition_batch, target_batch)
            else:
                source_condition, source_target = reference_loader.next()
                loss, _, alignment_loss, _, _ = dadiff.optimize_parameters(
                    condition_batch, target_batch,
                    source_condition.to(device).float(), source_target.to(device).float())

            total_loss += float(loss)
            total_alignment += float(alignment_loss)

        batches = max(len(loader), 1)
        print(f"[{direction}] epoch {epoch + 1}/{epochs} loss {total_loss / batches:.6f} "
              f"alignment {total_alignment / batches:.6f}")

    directory = models_dir(config, raw.name, tag)
    os.makedirs(directory, exist_ok=True)
    dadiff.save(os.path.join(directory, f'{direction}.pt'))
    print(f"[{direction}] model saved to {directory}")

    generate_images(config, raw, direction, dadiff, tag, device)


@torch.no_grad()
def generate_images(config, raw, direction, dadiff, tag, device):
    """Translate every patch of the scene and save the stitched image."""
    dataset = build_patch_dataset(config, raw, direction, raw.change_map, None, 'test')
    loader = DataLoader(dataset,
                        batch_size=config.get_path('dadiff.batch_size', 256),
                        shuffle=False, num_workers=config.get_path('dadiff.num_workers', 4))

    generated = []
    for condition_batch, target_batch in tqdm(loader, desc=f'{direction} sampling'):
        generated.append(dadiff.generate(condition_batch.to(device).float(),
                                         target_batch.to(device).float()).cpu())

    image = dataset.reconstruct(torch.cat(generated, dim=0)).numpy().transpose(1, 2, 0)

    output_dir = images_dir(config, raw.name, tag)
    name = 'rgb_generated.mat' if direction == 'rgb' else 'sar_generated.mat'
    save_mat(os.path.join(output_dir, name), 'img', image)
    print(f"[{direction}] translated image saved to {os.path.join(output_dir, name)}")


def main():
    parser = argparse.ArgumentParser(description="Train DADiff for one dataset and one stage")
    parser.add_argument('--config', required=True)
    parser.add_argument('--stage', choices=('pretrain', 'adapt', 'generate'), required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--round', type=int, default=None, help='Refinement round (adapt only)')
    parser.add_argument('--tag', default=None, help='Output tag (defaults to source or the round index)')
    parser.add_argument('--guidance-map', default=None,
                        help='Change map guiding the crop selection of the adapt stage.')
    parser.add_argument('--directions', nargs='+', choices=tuple(DIRECTIONS), default=list(DIRECTIONS))
    parser.add_argument('--model-dataset', default=None,
                        help='Dataset of the checkpoint to load in the generate stage '
                             '(defaults to the source dataset)')
    parser.add_argument('--model-tag', default='source',
                        help='Tag of the checkpoint to load in the generate stage')
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device(config.get_path('device', 'cuda:0'))
    set_seed(config.get_path('seed', 42))

    raw = load_raw_dataset(config, args.dataset)

    if args.stage == 'generate':
        # Translate a dataset with an already trained model, without training.
        model_dataset = args.model_dataset or config.require('task.source')
        tag = args.tag or 'source'
        for direction in args.directions:
            dadiff = build_dadiff(config, raw, direction, device, use_alignment=False)
            path = os.path.join(models_dir(config, model_dataset, args.model_tag), f'{direction}.pt')
            if not os.path.exists(path):
                raise FileNotFoundError(f"No model found at {path}")
            dadiff.netG.load_state_dict(torch.load(path, map_location=device)['netG'])
            print(f"[{direction}] model loaded from {path}")
            generate_images(config, raw, direction, dadiff, tag, device)
        return

    if args.stage == 'pretrain':
        tag = args.tag or 'source'
        epochs = config.get_path('dadiff.epochs.pretrain', 200)
        guidance_map, labelled, source_raw = raw.change_map, raw.labelled, None
    else:
        if args.round is None:
            raise SystemExit("--round is required for the adapt stage")
        if args.guidance_map is None:
            raise SystemExit("--guidance-map is required for the adapt stage")
        tag = args.tag or str(args.round)
        schedule = config.get_path('dadiff.epochs.adapt', [100, 50])
        epochs = schedule[args.round] if args.round < len(schedule) else schedule[-1]
        guidance_map, labelled = load_mat(args.guidance_map, 'pred'), raw.labelled
        source_raw = load_raw_dataset(config, config.require('task.source'))

    for direction in args.directions:
        train_direction(config, raw, direction, tag, guidance_map, labelled, epochs, device,
                        source_raw=source_raw)


if __name__ == '__main__':
    main()
