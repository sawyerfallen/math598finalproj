# Project Writeup Notes

## Question

This project asks whether explicit symbolic node-type embeddings help GPT-2 small solve algebra equations more accurately than a text-only fine-tuned baseline.

The baseline and structured models both start from GPT-2 small, which uses learned absolute positional embeddings. The structured model adds one learned embedding per coarse symbolic node type and sums that vector with the usual token embedding. The base model and the node-type embeddings are fine-tuned end to end.

## Final Experiment

The final dataset is solve-only and deduplicated before splitting. It mixes easy linear equations with harder solve-for-`x` equations involving parentheses, collecting like terms, variables on both sides, distribution, and simple quadratics with two real integer roots.

The training JSONL files contain only `prompt` and `output`. Metadata sidecars store `difficulty` and `solve_kind` for evaluation breakdowns.

Final split:

- `10,000` training examples
- `1,000` validation examples
- `1,000` test examples
- `0` duplicate prompt/output pairs shared across splits

## Results

| Metric | Baseline | Structured |
| --- | ---: | ---: |
| Overall exact match | 0.1830 | 0.1830 |
| Easy exact match | 0.2826 | 0.3370 |
| Hard exact match | 0.1729 | 0.1674 |

The structured model improved on the easy subset but did not improve overall. In this run, the added node-type signal was not enough to beat the text-only baseline on the harder deduplicated test distribution.

## Implementation Notes

The comparison script loads saved checkpoints only. It uses constrained greedy decoding, GPT-2-compatible position IDs for left-padded batches, and answer-span extraction so extra trailing text does not automatically invalidate an otherwise complete answer.

The structured run summaries verify that node-type labels are nontrivial, the node-type embedding table is trainable, and `structured_state.pt` contains the learned node-type weights.
