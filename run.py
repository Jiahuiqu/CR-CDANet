"""End-to-end driver of the CR-CDANet pipeline.

The pipeline consists of one preparation phase and a closed refinement loop:

1. ``train_dadiff --stage pretrain`` -- cross-modal diffusion on the source domain;
2. ``train_source``                  -- change detector on the source domain;
3. ``evaluate``                      -- preliminary change map of the target domain;
4. for each refinement round ``r``:
   a. ``train_dadiff --stage adapt``  -- DADiff fine-tuned on the target domain;
   b. ``train_round``                 -- DCCD with the labels of round ``r``;
   c. ``evaluate``                    -- change map used by round ``r + 1``.

    python run.py --config configs/shuguang2islandtown.yaml
    python run.py --config configs/shuguang2islandtown.yaml --from-round 2
"""

import argparse
import os
import subprocess
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config import load_config

HERE = os.path.dirname(os.path.abspath(__file__))


def run(command):
    print('\n$ ' + ' '.join(command), flush=True)
    subprocess.run(command, check=True, cwd=HERE)


def main():
    parser = argparse.ArgumentParser(description="Run the whole CR-CDANet pipeline")
    parser.add_argument('--config', required=True)
    parser.add_argument('--from-round', type=int, default=0,
                        help='First refinement round to run (earlier stages are assumed to be done)')
    parser.add_argument('--skip-preparation', action='store_true',
                        help='Skip steps 1-3 and start from the refinement loop')
    parser.add_argument('--directions', nargs='+', default=['rgb', 'sar'])
    args = parser.parse_args()

    config = load_config(args.config)
    task = config.require('task.name')
    source = config.require('task.source')
    target = config.require('task.target')
    rounds = config.get_path('task.rounds', len(config.get_path('cd.target_label_ratios', [])))

    python = sys.executable

    if not args.skip_preparation and args.from_round == 0:
        run([python, 'train_dadiff.py', '--config', args.config, '--stage', 'pretrain',
             '--dataset', source, '--directions', *args.directions])
        # Translate the target domain with the source model: it provides the
        # cross-modal images of the preliminary change map.
        run([python, 'train_dadiff.py', '--config', args.config, '--stage', 'generate',
             '--dataset', target, '--model-dataset', source, '--tag', 'source',
             '--directions', *args.directions])
        run([python, 'train_source.py', '--config', args.config, '--dataset', source])
        run([python, 'evaluate.py', '--config', args.config, '--tag', 'source',
             '--target-tag', 'source', '--predict-tag', 'preliminary'])

    for round_index in range(args.from_round, rounds):
        guidance_tag = 'preliminary' if round_index == 0 else f'round_{round_index - 1}'
        guidance = os.path.join(config.output_dir(task, 'predictions', target,
                                                  guidance_tag), 'predict_mat.mat')
        if not os.path.exists(guidance):
            raise FileNotFoundError(
                f"Missing the change map that guides round {round_index}: {guidance}")

        run([python, 'train_dadiff.py', '--config', args.config, '--stage', 'adapt',
             '--dataset', target, '--round', str(round_index),
             '--guidance-map', guidance, '--directions', *args.directions])
        run([python, 'train_round.py', '--config', args.config, '--round', str(round_index)])
        run([python, 'evaluate.py', '--config', args.config, '--tag', str(round_index),
             '--target-tag', str(round_index), '--predict-tag', f'round_{round_index}'])

    print(f"\nPipeline finished. Results of each round are under "
          f"{config.output_dir(task, 'predictions', target)}")


if __name__ == '__main__':
    main()
