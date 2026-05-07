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

7. `src/algebra_generation.py` is the decoder so when evaluating these we can accurately decide if these models worked.

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
