# CR-CDANet

Official implementation of

> **Cycle-Refined Cross-Modal Diffusion-Based Domain Adaptation Network for
> Multimodal Remote Sensing Images Change Detection**
> Huabin Wu, Jiahui Qu, Wenqian Dong, Yunsong Li.
> *IEEE Transactions on Circuits and Systems for Video Technology.*

This repository contains the complete pipeline of the method for the
Shuguang ↔ Island Town cross-scene experiments (SAR / RGB pairs):

* **DADiff** — a domain-adaptive conditional diffusion model that performs
  bidirectional cross-modal translation, so that each modality becomes available
  at both acquisition dates;
* **DCCD** — a domain adaptation and co-training based change detector built on
  the AT-Mamba encoder/decoder, with a multi-scale joint domain adaptation loss
  and two alternating classifiers;
* **Cycle refinement** — a closed loop in which high-confidence pseudo labels are
  selected from the previous prediction and fed back to the next training round.

Both released tasks are cross-scene experiments with 1% of the target labels
revealed:

| configuration | source domain | target domain |
|---------------|---------------|---------------|
| `configs/shuguang2islandtown.yaml` | Shuguang | Island Town |
| `configs/islandtown2shuguang.yaml` | Island Town | Shuguang |

## Installation

```
conda create -n crcdanet python=3.10
conda activate crcdanet
pip install -r requirements.txt
```

`mamba-ssm` requires a CUDA toolchain matching the installed PyTorch build; see
the [mamba](https://github.com/state-spaces/mamba) repository for details. A GPU
is required both for training and for inference, because the selective scan
kernel is GPU only.

## Data

The datasets are not distributed here.  Put the files of Shuguang and Island
Town under `datasets/` as described in [`datasets/README.md`](datasets/README.md),
or point `data.root` of the configuration file to their location.

Quick sanity check of the implementation (CPU, no data needed, a few seconds):

```
python tools/checks.py
```

## Quick start

```
python run.py --config configs/shuguang2islandtown.yaml
```

`run.py` executes the whole pipeline and writes every intermediate result under
`outputs/<task>/`:

```
outputs/<task>/
├── dadiff/<dataset>/<round>/          # translated images of one domain and round
├── dadiff_models/<dataset>/<tag>/     # DADiff checkpoints (rgb.pt, sar.pt)
├── cd/<tag>/checkpoint.pt             # change detection checkpoints
└── predictions/<dataset>/<tag>/predict_mat.mat
```

The `predict_mat.mat` files hold the change map of every round; the map of the
last round is the final result. Metrics are also printed by `evaluate.py`.

## Pipeline in detail

Each stage can be run on its own.

**1. DADiff on the source domain** (200 epochs)

```
python train_dadiff.py --config configs/shuguang2islandtown.yaml \
    --stage pretrain --dataset shuguang
```

Both translation directions are trained. The training crops are restricted to
the high-confidence unchanged regions of the change map, as the paper prescribes:
a crop is kept only when all of its pixels pass the confidence test of
`src/refinement/pseudo_label.py`. The translated images are written to
`dadiff/<dataset>/source/`.

**2. Translate the target domain with the source model**

```
python train_dadiff.py --config configs/shuguang2islandtown.yaml \
    --stage generate --dataset islandtown --model-dataset shuguang --tag source
```

No training happens here: the model trained on the source domain is applied to
the target domain. These zero-shot translations are what the preliminary change
map is computed from.

**3. Change detector on the source domain** (fully supervised, 100 epochs)

```
python train_source.py --config configs/shuguang2islandtown.yaml
```

It provides the initial state of the source branch and the preliminary change
map of the target domain.

**4. Preliminary change map of the target domain**

```
python evaluate.py --config configs/shuguang2islandtown.yaml \
    --tag source --target-tag source --predict-tag preliminary
```

The source-domain detector is applied to the target domain. This map weakly
guides the crop selection of the first adaptation and never uses the target
labels.

**5. Refinement rounds** — repeat for `r = 0 … 4`

```
python train_dadiff.py --config configs/shuguang2islandtown.yaml --stage adapt \
    --dataset islandtown --round 0 --guidance-map outputs/<task>/predictions/islandtown/preliminary/predict_mat.mat

python train_round.py  --config configs/shuguang2islandtown.yaml --round 0
python evaluate.py     --config configs/shuguang2islandtown.yaml --tag 0 --target-tag 0 --predict-tag round_0
```

Round `r` trains with `target_label_ratios[r]` of the labelled pixels of each
class: 1% of ground truth for round 0, then 5%, 10%, 15% and 20%, the additional
supervision coming from the high-confidence pseudo labels of round `r - 1`.

Use `--round`/`--target-tag` to reuse the translations of an earlier stage, for
instance to ablate the domain adaptation of DADiff by keeping `--target-tag source`
for every round.

## Configuration

`configs/base.yaml` documents every hyper-parameter. The entries worth knowing:

| key | meaning |
|-----|---------|
| `dadiff.timesteps` | total diffusion steps `T` (2000) |
| `dadiff.sample_steps` | denoising iterations at inference time (100) |
| `dadiff.loss_type` | diffusion loss, `l2` |
| `dadiff.alphas` | MMD weight of the three encoder scales (0.3, 0.5, 0.7) |
| `cd.source_label_ratio` | fraction of the source labels used (1.0 = fully supervised) |
| `cd.target_label_ratios` | label ratio of each refinement round |
| `cd.alphas` | MMD / CORAL weight per scale of the change detector |
| `refinement.tau1`, `tau2` | agreement of the first and second expansion ring |
| `refinement.tau_b` | agreement of the boundary sub-regions |

## Reported results

Overall accuracy / Kappa / F1 with 1% of the target labels revealed:

| task | OA | KC | F1 |
|------|----|----|----|
| Shuguang → Island Town | 0.9666 | 0.6090 | 0.6264 |
| Island Town → Shuguang | 0.9931 | 0.9227 | 0.9263 |

## Repository layout

```
src/
├── models/at_mamba.py           AT-Mamba encoder/decoder for change detection
├── models/diffusion_backbone.py time-conditioned backbone of DADiff
├── models/classifier.py         decision-level classifier
├── losses/                      MMD, CORAL and their multi-scale fusion
├── refinement/pseudo_label.py   intra-region and boundary confidence scores
├── data/                        raw datasets, cross-modal pairing, patch datasets
├── dadiff.py                    Gaussian diffusion and DADiff training interface
├── engine.py                    shared model/checkpoint helpers
├── metrics.py                   OA, Kappa, F1, precision, recall
└── utils.py                     seeding, I/O, padding, normalisation
train_dadiff.py                  stage 1 - DADiff pre-training and adaptation
train_source.py                  stage 2 - change detector on the source domain
train_round.py                   stage 3a - one DCCD refinement round
evaluate.py                      stage 3b - dense prediction and metrics
run.py                           end-to-end driver
```

## Implementation notes

* The network predicts the clean target image directly instead of the injected
  noise (Sec. III-B), and is conditioned on `sqrt(alpha_bar)` rather than on the
  time step index.
* The inference loop runs over the last `dadiff.sample_steps` time steps
  (100 by default, as reported in Table II of the paper).
* The diffusion loss is computed on the whole `16 x 16` crop with an L2 loss;
  the crops themselves are drawn from high-confidence unchanged regions.
* Batch sizes are 256 for DADiff and 128 for the change detector.
* Change maps hold `1 = changed` and `0 = unchanged`; both datasets are fully
  labelled, so every pixel takes part in training and in the metrics.

## Citation

```bibtex
@article{wu2026crcdanet,
  title   = {Cycle-Refined Cross-Modal Diffusion-Based Domain Adaptation Network
             for Multimodal Remote Sensing Images Change Detection},
  author  = {Wu, Huabin and Qu, Jiahui and Dong, Wenqian and Li, Yunsong},
  journal = {IEEE Transactions on Circuits and Systems for Video Technology},
  year    = {2026}
}
```
