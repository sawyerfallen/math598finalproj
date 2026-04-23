Map of code:

1. `main.py` runs both models on the same testing set, computes the accuracy, and writes it to a json and txt

2. `src/train_baseline.py` fine-tunes the normal pythia model, without my additional labelings

3. `src/train_structured.py` trains my embedding on the dataset with the labelings

4. `src/structured_model.py` adds the learned embedding to the pythia model

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

Overall the structured model goes from `parser`, which turns the prompt text into the tokens and the labels, to `node_types` which maps these to ints, to `structured_dataset` which tokenizes it, to `structured_model` which adds the typed embeddings to the standard embeddings, to `train_structured` which trains the experimental model.

8. `dataset_generator` just makes a dataset using sympy to ensure it's correct.

9. `data` folder contains the train/test/validation splits from the generator.

10. `artifacts` has the training runs and experiment results.

11. Writeup is in `WRITEUP.md` It isn't the formal writeup yet. 

12. 'language.md' currently just has a few examples with prompts to show what I am trying to do.

Questions:

What are some tips with prompting coding agents?