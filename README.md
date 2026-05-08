# Structured GPT-2 Algebra Experiment

This project tests whether giving a language model explicit symbolic structure helps it solve algebra problems more reliably. The research question is: **does adding coarse node-type information for algebraic symbols improve a causal language model beyond ordinary text-only fine-tuning?**

The experiment compares two GPT-2 small fine-tuning setups on the same synthetic data:

- **Baseline:** GPT-2 small fine-tuned on plain `prompt` / `output` text.
- **Structured:** the same GPT-2 small model, but prompt tokens also receive learned node-type embeddings such as `TASK`, `VARIABLE`, `CONSTANT`, `ADD`, `MUL`, `POW`, `EQUALITY`, and `PUNCT`.

The final experiment is solve-only symbolic algebra. Every prompt asks the model to solve for `x`:

```text
solve 2*x + 3 = 7 for x =>
```

The target output is a canonical answer such as:

```text
x = 2
```

For quadratics with two real integer roots, the output uses sorted roots:

```text
x = -1 or x = 3
```

The dataset mixes easy linear equations with harder equations that require parentheses handling, distribution, collecting like terms, moving `x` terms across both sides, and solving simple factorable quadratics. Evaluation uses accuracy against the canonical extracted answer as the main metric.

## Final Results

Latest run:

- Dataset: mixed solve-only algebra examples
- Baseline run: GPT-2 small text-only fine-tuning
- Structured run: GPT-2 small with node-type embeddings
- Comparison: saved baseline-vs-structured evaluation outputs
- Plots: `artifacts/plots/solve_mixed_10000_bundle`

Dataset summary:

- Raw generated examples: `20,968`
- Examples used: `12,000`
- Final split sizes: `10,000` train, `1,000` validation, `1,000` test

Held-out test accuracy:

| Metric | Baseline | Structured |
| --- | ---: | ---: |
| Overall accuracy | 0.1830 | 0.1830 |
| Easy accuracy | 0.2826 | 0.3370 |
| Hard accuracy | 0.1729 | 0.1674 |

## How The Pipeline Fits Together

1. **Dataset generation:** `src/dataset_generator.py` creates synthetic solve-for-`x` prompt/output pairs, splits them into train/validation/test JSONL files, and writes metadata sidecars with `difficulty` and `solve_kind`.

2. **Prompt parsing:** `src/parser.py` reads the prompt text and emits symbolic prompt tokens plus node-type labels. For example, `solve`, `for`, `x`, integer constants, operators, equality, parentheses, and the `=>` marker receive coarse labels from `src/node_types.py`.

3. **Tokenizer alignment:** `src/structured_dataset.py` maps parser-level symbolic tokens onto GPT-2 tokenizer pieces using tokenizer offset mappings. If a GPT-2 token cleanly overlaps one symbolic token, it receives that symbolic node type. If a tokenizer piece merges multiple symbolic types, it is labeled `OTHER`.

4. **Structured dataset creation:** `StructuredJsonlDataset` builds the full training text as `prompt + output + EOS`. It creates normal `input_ids`, `attention_mask`, and `labels`, then adds a parallel `node_type_ids` tensor. Prompt tokens receive aligned node-type ids; answer tokens and padding use `OTHER`. The prompt region is still masked with `-100`, so training loss is applied only to answer tokens.

5. **Baseline training:** `src/train_baseline.py` loads the same JSONL files, tokenizes prompt/output text, masks prompt labels, and fine-tunes GPT-2 small without node-type inputs.

6. **Structured training:** `src/train_structured.py` uses `StructuredJsonlDataset` and `StructuredCollator`. It fine-tunes GPT-2 small together with the added node-type embedding table and saves both the base model files and `structured_state.pt`.

7. **Structured model:** `src/structured_model.py` wraps `AutoModelForCausalLM`. On each forward pass, it looks up GPT-2 token embeddings and node-type embeddings, sums them position-wise, and passes the result into the base causal LM through `inputs_embeds`.

8. **Generation helpers:** `src/algebra_generation.py` performs constrained greedy decoding. It limits generated tokens to algebra-relevant characters, handles GPT-2 left-padding position IDs, stops once a complete solve answer appears, and extracts the first valid answer span for scoring.

9. **Comparison:** `src/main.py` loads saved baseline and structured checkpoints, runs both on the same test set, scores accuracy, verifies the structured checkpoint contains node-type weights, and writes JSON, TXT, and per-sample JSONL outputs.

10. **Plotting:** `src/plot_training_curves.py`, `src/plot_batch_losses.py`, and `src/plot_comparison_accuracy.py` turn saved metrics and predictions into loss, accuracy, difficulty, solve-kind, and accuracy-split plots.

## Structured Dataset Construction

The structured model does not see different text from the baseline. Both models train on the same JSONL prompt/output pairs. The difference is that the structured path adds one extra tensor:

- The dataset generator writes examples like `{"prompt": "solve 2*x + 3 = 7 for x =>", "output": "x = 2"}`.
- The parser labels prompt-level symbolic pieces, for example `solve -> TASK`, `2 -> CONSTANT`, `* -> MUL`, `x -> VARIABLE`, `= -> EQUALITY`.
- `structured_dataset.py` aligns those symbolic labels to GPT-2 tokenizer pieces.
- The resulting training item contains `input_ids`, `attention_mask`, prompt-masked `labels`, and aligned `node_type_ids`.
- `structured_model.py` consumes both `input_ids` and `node_type_ids`; it adds the learned node-type embedding to the normal token embedding before calling GPT-2.
- The answer tokens are still the only tokens that contribute to the causal LM loss, matching the baseline prompt-masking setup.

## `src/` Codebase Map

- `src/__init__.py` marks the source directory as the importable package used by the CLI scripts.
- `src/dataset_generator.py` generates the final solve-only dataset, metadata sidecars, and train/validation/test splits.
- `src/node_types.py` defines the node-type vocabulary shared by the parser, dataset, and model.
- `src/parser.py` tokenizes solve prompts into symbolic tokens and node-type names.
- `src/structured_dataset.py` aligns node types to tokenizer pieces and builds structured training batches.
- `src/structured_model.py` defines the GPT-2 wrapper with learned node-type embeddings.
- `src/train_baseline.py` trains the text-only GPT-2 baseline with prompt-masked causal LM loss.
- `src/train_structured.py` trains the structured GPT-2 model and logs node-type / trainable-parameter verification.
- `src/algebra_generation.py` contains constrained decoding, answer extraction, and structured generation helpers.
- `src/main.py` is the saved-checkpoint comparison script.
- `src/plot_training_curves.py` plots train/validation loss curves.
- `src/plot_batch_losses.py` plots per-batch training loss.
- `src/plot_comparison_accuracy.py` plots overall, easy/hard, solve-kind, and accuracy-split comparison results.
- `src/utils.py` contains shared logging, padding, parameter-counting, device-transfer, and loss helpers.

## Dataset Format

Training, validation, and test JSONL files contain only model-visible fields:

```json
{"prompt": "solve 2*x + 3 = 7 for x =>", "output": "x = 2"}
```

Metadata files such as `test_metadata.jsonl` are sidecars used only for grouped evaluation and plotting. They store fields like `difficulty` and `solve_kind`; they are not included in the training examples.

## Commands

Install dependencies:

```powershell
uv sync
```

The checked-in artifact files contain the final run. The CLI entry points are:

- `uv run generate-dataset`
- `uv run train-baseline`
- `uv run train-structured`
- `uv run compare-models`
- `uv run plot-training-curves`
- `uv run plot-batch-losses`
- `uv run plot-comparison-accuracy`

## Evaluation Notes

Comparison always loads saved checkpoints; it does not retrain. Generation uses constrained greedy decoding, GPT-2-compatible left-padding position IDs, and first-answer-span extraction so a correct answer is not unfairly marked wrong only because extra text follows it.

The structured training summary records trainable parameter counts, node-type label usage, and checkpoint verification. The comparison summary also checks that `structured_state.pt` contains a nonzero `node_type_embedding`.
