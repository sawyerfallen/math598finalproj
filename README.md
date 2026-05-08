Map of code:

1. `src/main.py` runs both models on the same testing set, computes the accuracy, and writes it to a json and txt

2. `src/train_baseline.py` fine-tunes the normal GPT-2 small model, without my additional labelings

3. `src/train_structured.py` fine-tunes GPT-2 small on the same dataset with my additional labelings

4. `src/structured_model.py` adds the learned embedding to the GPT-2 small model

5. `src/structured_dataset.py` loads the dataset and builds the training text from the dataset, and also tokenizes it and adds the labels as another vector.

6. `src/parser.py` creates the labels from the prompt:
   - task words become `TASK`
   - `x`, `y`, `z` become `VARIABLE`
   - integers become `CONSTANT`
   - `+` and `-` in additive position become `ADD`
   - `*` becomes `MUL`
   - `^` or `**` become `POW`
   - `=` becomes `EQUALITY`
   - parentheses and `=>` become `PUNCT`
   - everything else falls back to `OTHER`

7. `src/algebra_generation.py` is the constrained decoder. It handles GPT-2 position ids for left-padded batches and can stop once a complete task answer appears.

8. `src/utils.py` has shared helper code for logging, summaries, parameter counts, tokenizer padding, and moving batches to a device.

Overall the structured model goes from `parser`, which turns the prompt text into the tokens and the labels, to `node_types` which maps these to ints, to `structured_dataset` which tokenizes it, to `structured_model` which adds the typed embeddings to the standard embeddings, to `train_structured` which fine-tunes GPT-2 small and the typed embeddings together. `train_structured` also has `--freeze-base` if I want an ablation where only the added typed embeddings train.

9. `src/dataset_generator.py` just makes a dataset using sympy to ensure it's correct.

10. `data` folder contains the train/test/validation splits from the generator.

11. `artifacts` has the training runs and experiment results.

12. Writeup is in `WRITEUP.md` It isn't the formal writeup yet.

13. 'language.md' currently just has a few examples with prompts to show what I am trying to do.

Questions:

What are some tips with prompting coding agents?

Pipeline commands:

Generate the default 6000-example dataset split:
`uv run generate-dataset`

Generate a custom dataset split:
`uv run generate-dataset --dataset-size 12000 --output-dir data`

Train the baseline model and save `final-model`, metrics, per-sample test losses, and a summary:
`uv run train-baseline --model-name gpt2 --experiment-name gpt2-small-baseline --output-dir artifacts/models_training_info/gpt2-small-baseline --epochs 1`

Train the structured model with full fine-tuning and save `final-model`, metrics, per-sample test losses, and a summary:
`uv run train-structured --model-name gpt2 --experiment-name gpt2-small-structured-node-types --output-dir artifacts/models_training_info/gpt2-small-structured-node-types --epochs 1`

Compare a saved baseline checkpoint against a saved structured checkpoint without retraining:
`uv run compare-models --baseline-checkpoint artifacts/models_training_info/gpt2-small-baseline/final-model --structured-checkpoint artifacts/models_training_info/gpt2-small-structured-node-types/final-model --test-path data/test.jsonl --output-path artifacts/comparisons/gpt2-small-baseline-vs-structured.json --text-output-path artifacts/comparisons/gpt2-small-baseline-vs-structured.txt --per-sample-output-path artifacts/comparisons/gpt2-small-baseline-vs-structured-per-sample.jsonl`

Plot the saved per-sample test losses:
`uv run plot-test-losses artifacts/models_training_info/gpt2-small-baseline/test_sample_losses.jsonl`

Plot training loss over batches for both one-epoch runs:
`uv run plot-batch-losses artifacts/models_training_info/gpt2-small-baseline/metrics.jsonl artifacts/models_training_info/gpt2-small-structured-node-types/metrics.jsonl --labels baseline structured --output-path artifacts/plots/gpt2-small-batch-losses.png`

Plot comparison accuracies from the saved per-sample JSONL:
`uv run plot-comparison-accuracy artifacts/comparisons/gpt2-small-baseline-vs-structured-per-sample.jsonl --prefix gpt2-small-baseline-vs-structured`

Solve-only narrow-task pipeline:

Generate solve-only train/validation/test splits:
`uv run generate-dataset --dataset-size 1200 --tasks solve --output-dir data/solve_only --seed 42`

Train the solve-only baseline and save best/final checkpoints:
`uv run train-baseline --model-name gpt2 --experiment-name gpt2-small-baseline-solve-only --train-path data/solve_only/train.jsonl --val-path data/solve_only/val.jsonl --test-path data/solve_only/test.jsonl --output-dir artifacts/models_training_info/gpt2-small-baseline-solve-only --epochs 3 --batch-size 16 --eval-batch-size 16 --lr 1e-4 --save-best-checkpoint`

Train the solve-only structured model with the same setup:
`uv run train-structured --model-name gpt2 --experiment-name gpt2-small-structured-solve-only --train-path data/solve_only/train.jsonl --val-path data/solve_only/val.jsonl --test-path data/solve_only/test.jsonl --output-dir artifacts/models_training_info/gpt2-small-structured-solve-only --epochs 3 --batch-size 16 --eval-batch-size 16 --lr 1e-4 --save-best-checkpoint`

Compare saved solve-only checkpoints without retraining:
`uv run compare-models --baseline-checkpoint artifacts/models_training_info/gpt2-small-baseline-solve-only/best-epoch-3 --structured-checkpoint artifacts/models_training_info/gpt2-small-structured-solve-only/best-epoch-3 --test-path data/solve_only/test.jsonl --batch-size 32 --max-new-tokens 8 --output-path artifacts/comparisons/gpt2-small-solve-only-baseline-vs-structured.json --text-output-path artifacts/comparisons/gpt2-small-solve-only-baseline-vs-structured.txt --per-sample-output-path artifacts/comparisons/gpt2-small-solve-only-baseline-vs-structured-per-sample.jsonl`

Plot solve-only train/validation loss:
`uv run plot-training-curves artifacts/models_training_info/gpt2-small-baseline-solve-only/metrics.jsonl artifacts/models_training_info/gpt2-small-structured-solve-only/metrics.jsonl --labels baseline structured --output-path artifacts/plots/gpt2-small-solve-only-training-curves.png --title "Solve-Only Train and Validation Loss"`

Plot solve-only comparison accuracies and generation breakdown:
`uv run plot-comparison-accuracy artifacts/comparisons/gpt2-small-solve-only-baseline-vs-structured-per-sample.jsonl --prefix gpt2-small-solve-only-baseline-vs-structured`

Harder solve-only pipeline:

Generate harder solve-only train/validation/test splits. This keeps the JSONL format as `prompt` and `output`, solves only for `x`, and includes parenthesized linear equations, collecting like terms, `x` on both sides, distribution, and simple quadratics with two real integer roots:
`uv run generate-dataset --tasks solve --solve-difficulty hard --train-size 5000 --val-size 500 --test-size 500 --output-dir data/solve_hard --seed 42`

Train the harder solve-only baseline and save best/final checkpoints:
`uv run train-baseline --model-name gpt2 --experiment-name gpt2-small-baseline-solve-hard --train-path data/solve_hard/train.jsonl --val-path data/solve_hard/val.jsonl --test-path data/solve_hard/test.jsonl --output-dir artifacts/models_training_info/gpt2-small-baseline-solve-hard --epochs 2 --batch-size 32 --eval-batch-size 32 --lr 1e-4 --save-best-checkpoint`

Train the harder solve-only structured model on the same splits:
`uv run train-structured --model-name gpt2 --experiment-name gpt2-small-structured-solve-hard --train-path data/solve_hard/train.jsonl --val-path data/solve_hard/val.jsonl --test-path data/solve_hard/test.jsonl --output-dir artifacts/models_training_info/gpt2-small-structured-solve-hard --epochs 2 --batch-size 32 --eval-batch-size 32 --lr 1e-4 --save-best-checkpoint`

Compare saved harder solve-only checkpoints without retraining:
`uv run compare-models --baseline-checkpoint artifacts/models_training_info/gpt2-small-baseline-solve-hard/best-epoch-2 --structured-checkpoint artifacts/models_training_info/gpt2-small-structured-solve-hard/best-epoch-2 --test-path data/solve_hard/test.jsonl --batch-size 32 --max-new-tokens 20 --output-path artifacts/comparisons/gpt2-small-solve-hard-baseline-vs-structured.json --text-output-path artifacts/comparisons/gpt2-small-solve-hard-baseline-vs-structured.txt --per-sample-output-path artifacts/comparisons/gpt2-small-solve-hard-baseline-vs-structured-per-sample.jsonl`

Plot harder solve-only train/validation loss:
`uv run plot-training-curves artifacts/models_training_info/gpt2-small-baseline-solve-hard/metrics.jsonl artifacts/models_training_info/gpt2-small-structured-solve-hard/metrics.jsonl --labels baseline structured --output-path artifacts/plots/gpt2-small-solve-hard-training-curves.png --title "Hard Solve-Only Train and Validation Loss"`

Plot harder solve-only comparison accuracies and generation breakdown:
`uv run plot-comparison-accuracy artifacts/comparisons/gpt2-small-solve-hard-baseline-vs-structured-per-sample.jsonl --prefix gpt2-small-solve-hard-baseline-vs-structured`

Plot harder solve-only per-sample test losses:
`uv run plot-test-losses artifacts/models_training_info/gpt2-small-baseline-solve-hard/test_sample_losses.jsonl --output-path artifacts/plots/gpt2-small-solve-hard-baseline-test-losses.png --title "Baseline Hard Solve Per-Sample Test Loss"`

`uv run plot-test-losses artifacts/models_training_info/gpt2-small-structured-solve-hard/test_sample_losses.jsonl --output-path artifacts/plots/gpt2-small-solve-hard-structured-test-losses.png --title "Structured Hard Solve Per-Sample Test Loss"`

Mixed easy/hard solve-only pipeline:

Generate mixed solve-only splits with 10000 train, 1000 validation, and 1000 test examples. Training JSONL files stay clean with only `prompt` and `output`; the generator also writes `*_metadata.jsonl` sidecars with `difficulty` and `solve_kind` for grouped evaluation:
`uv run generate-dataset --tasks solve --solve-difficulty mixed --train-size 10000 --val-size 1000 --test-size 1000 --output-dir data/solve_mixed_10000 --seed 43`

Train the mixed solve-only baseline and save best/final checkpoints plus trainable-parameter verification:
`uv run train-baseline --model-name gpt2 --experiment-name gpt2-small-baseline-solve-mixed-10000 --train-path data/solve_mixed_10000/train.jsonl --val-path data/solve_mixed_10000/val.jsonl --test-path data/solve_mixed_10000/test.jsonl --output-dir artifacts/models_training_info/gpt2-small-baseline-solve-mixed-10000 --epochs 1 --batch-size 32 --eval-batch-size 32 --lr 1e-4 --save-best-checkpoint`

Train the mixed solve-only structured model on the same splits. The summary records trainable parameters, node-type label usage, and structured checkpoint checks:
`uv run train-structured --model-name gpt2 --experiment-name gpt2-small-structured-solve-mixed-10000 --train-path data/solve_mixed_10000/train.jsonl --val-path data/solve_mixed_10000/val.jsonl --test-path data/solve_mixed_10000/test.jsonl --output-dir artifacts/models_training_info/gpt2-small-structured-solve-mixed-10000 --epochs 1 --batch-size 32 --eval-batch-size 32 --lr 1e-4 --save-best-checkpoint`

Compare saved mixed solve-only checkpoints without retraining, including easy vs hard breakdown:
`uv run compare-models --baseline-checkpoint artifacts/models_training_info/gpt2-small-baseline-solve-mixed-10000/best-epoch-1 --structured-checkpoint artifacts/models_training_info/gpt2-small-structured-solve-mixed-10000/best-epoch-1 --test-path data/solve_mixed_10000/test.jsonl --metadata-path data/solve_mixed_10000/test_metadata.jsonl --batch-size 32 --max-new-tokens 20 --output-path artifacts/comparisons/gpt2-small-solve-mixed-10000-baseline-vs-structured.json --text-output-path artifacts/comparisons/gpt2-small-solve-mixed-10000-baseline-vs-structured.txt --per-sample-output-path artifacts/comparisons/gpt2-small-solve-mixed-10000-baseline-vs-structured-per-sample.jsonl`

Plot mixed solve-only train/validation loss:
`uv run plot-training-curves artifacts/models_training_info/gpt2-small-baseline-solve-mixed-10000/metrics.jsonl artifacts/models_training_info/gpt2-small-structured-solve-mixed-10000/metrics.jsonl --labels baseline structured --output-path artifacts/plots/gpt2-small-solve-mixed-10000-training-curves.png --title "Mixed Solve-Only Train and Validation Loss"`

Plot mixed solve-only overall accuracy, easy/hard accuracy, and generation breakdown:
`uv run plot-comparison-accuracy artifacts/comparisons/gpt2-small-solve-mixed-10000-baseline-vs-structured-per-sample.jsonl --prefix gpt2-small-solve-mixed-10000-baseline-vs-structured`

Plot mixed solve-only per-sample test losses:
`uv run plot-test-losses artifacts/models_training_info/gpt2-small-baseline-solve-mixed-10000/test_sample_losses.jsonl --output-path artifacts/plots/gpt2-small-solve-mixed-10000-baseline-test-losses.png --title "Baseline Mixed Solve Per-Sample Test Loss"`

`uv run plot-test-losses artifacts/models_training_info/gpt2-small-structured-solve-mixed-10000/test_sample_losses.jsonl --output-path artifacts/plots/gpt2-small-solve-mixed-10000-structured-test-losses.png --title "Structured Mixed Solve Per-Sample Test Loss"`

Deduplicated mixed easy/hard solve-only pipeline:

Generate mixed solve-only splits after deduplicating by `prompt` and `output` before train/validation/test splitting. The generator writes clean training JSONL files plus `*_metadata.jsonl` sidecars and `generation_summary.json` / `generation_summary.txt` with raw, unique, duplicate-removed, split-size, and cross-split duplicate counts:
`uv run generate-dataset --tasks solve --solve-difficulty mixed --train-size 10000 --val-size 1000 --test-size 1000 --deduplicate-before-split --max-raw-examples 100000 --dedup-batch-size 5000 --output-dir data/solve_mixed_10000_dedup --seed 45`

Train the deduplicated mixed solve-only baseline and save best/final checkpoints plus trainable-parameter verification:
`uv run train-baseline --model-name gpt2 --experiment-name gpt2-small-baseline-solve-mixed-10000-dedup --train-path data/solve_mixed_10000_dedup/train.jsonl --val-path data/solve_mixed_10000_dedup/val.jsonl --test-path data/solve_mixed_10000_dedup/test.jsonl --output-dir artifacts/models_training_info/gpt2-small-baseline-solve-mixed-10000-dedup --epochs 1 --batch-size 32 --eval-batch-size 32 --lr 1e-4 --save-best-checkpoint`

Train the deduplicated mixed solve-only structured model on the same splits. The summary records trainable parameters, nontrivial node-type label usage, and structured checkpoint checks:
`uv run train-structured --model-name gpt2 --experiment-name gpt2-small-structured-solve-mixed-10000-dedup --train-path data/solve_mixed_10000_dedup/train.jsonl --val-path data/solve_mixed_10000_dedup/val.jsonl --test-path data/solve_mixed_10000_dedup/test.jsonl --output-dir artifacts/models_training_info/gpt2-small-structured-solve-mixed-10000-dedup --epochs 1 --batch-size 32 --eval-batch-size 32 --lr 1e-4 --save-best-checkpoint`

Compare saved deduplicated checkpoints without retraining, including overall and easy/hard metrics:
`uv run compare-models --baseline-checkpoint artifacts/models_training_info/gpt2-small-baseline-solve-mixed-10000-dedup/best-epoch-1 --structured-checkpoint artifacts/models_training_info/gpt2-small-structured-solve-mixed-10000-dedup/best-epoch-1 --test-path data/solve_mixed_10000_dedup/test.jsonl --metadata-path data/solve_mixed_10000_dedup/test_metadata.jsonl --batch-size 32 --max-new-tokens 20 --output-path artifacts/comparisons/gpt2-small-solve-mixed-10000-dedup-baseline-vs-structured.json --text-output-path artifacts/comparisons/gpt2-small-solve-mixed-10000-dedup-baseline-vs-structured.txt --per-sample-output-path artifacts/comparisons/gpt2-small-solve-mixed-10000-dedup-baseline-vs-structured-per-sample.jsonl`

Create the dedicated deduplicated plot bundle:
`uv run plot-training-curves artifacts/models_training_info/gpt2-small-baseline-solve-mixed-10000-dedup/metrics.jsonl artifacts/models_training_info/gpt2-small-structured-solve-mixed-10000-dedup/metrics.jsonl --labels baseline structured --output-path artifacts/plots/solve_mixed_10000_dedup_bundle/training-curves.png --title "Deduplicated Mixed Solve-Only Train and Validation Loss"`

`uv run plot-batch-losses artifacts/models_training_info/gpt2-small-baseline-solve-mixed-10000-dedup/metrics.jsonl artifacts/models_training_info/gpt2-small-structured-solve-mixed-10000-dedup/metrics.jsonl --labels baseline structured --output-path artifacts/plots/solve_mixed_10000_dedup_bundle/batch-losses.png --title "Deduplicated Mixed Solve-Only Training Loss by Batch"`

`uv run plot-comparison-accuracy artifacts/comparisons/gpt2-small-solve-mixed-10000-dedup-baseline-vs-structured-per-sample.jsonl --output-dir artifacts/plots/solve_mixed_10000_dedup_bundle --prefix accuracy`

`uv run plot-test-losses artifacts/models_training_info/gpt2-small-baseline-solve-mixed-10000-dedup/test_sample_losses.jsonl --output-path artifacts/plots/solve_mixed_10000_dedup_bundle/baseline-test-losses.png --title "Baseline Deduplicated Mixed Solve Per-Sample Test Loss"`

`uv run plot-test-losses artifacts/models_training_info/gpt2-small-structured-solve-mixed-10000-dedup/test_sample_losses.jsonl --output-path artifacts/plots/solve_mixed_10000_dedup_bundle/structured-test-losses.png --title "Structured Deduplicated Mixed Solve Per-Sample Test Loss"`
