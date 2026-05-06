- What is the question you hope to answer? How does this relate to existing literature?

I want to test whether adding explicit structural encodings to mathematical symbols helps a language model reason more accurately about symbolic algebra than a standard token-only language model. My original proposal framed this broadly in terms of sets, functions, and operators, but I narrowed the project to symbolic algebra so the setting would be easier to experiment with. This connects to prior work on tree-based and structure-aware representations of mathematical language, but my approach differs by assigning explicit symbolic labels such as variable, constant, operator, and equality rather than relying only on contextual or unsupervised structure.

- How do you plan to answer it? How does your proposed method relate to existing ones?

I compare two versions of GPT-2 small on the same synthetic algebra dataset: a baseline trained only on prompt-output text pairs, and a structured model that adds learned node-type embeddings to the token embeddings. I switched from Pythia-70M to GPT-2 small because GPT-2 uses learned absolute positional embeddings rather than rotary positional embeddings. The structured model is now fine-tuned on the same training set as the baseline model, with both the GPT-2 weights and the added node-type embeddings updated during training. The parser assigns symbolic labels to prompt tokens, and those labels are aligned to tokenizer pieces before being passed into the model. This is related to structure-aware methods in the literature, but my implementation is a simpler typed-embedding approach rather than a full tree encoder.

- What experiments have you run, what results have you gotten? How did they make you change your mind?

I (mostly Codex) have already built the dataset pipeline, baseline trainer, structured trainer, and comparison script, and I ran a direct evaluation on 500 examples.

| Metric | Baseline | Structured | Delta |
| --- | --- | --- | --- |
| Exact Match Accuracy | 0.3980 | 0.3260 | -0.0720 |
| Symbolic Accuracy | 0.4100 | 0.3260 | -0.0840 |

The previous structured run likely did worse because it only trained the added node-type embeddings while the underlying language model stayed frozen. The current setup fixes that comparison by fine-tuning the structured model on the same dataset as the baseline.


- What experiments do you have left to run?

I would like to possibly train on more examples. My laptop currently takes about 10 minutes to train on the 5000 dataset example, and the accuracy should hopefully go up for both models when I train on more examples.

I would also like to compare results based on the type of task I am asking the models to do. 

- How will the answer to your question depend on those experiments?

If better-trained models still show no difference, then the conclusion will be that simple labelled embeddings do not improve symbolic algebra performance in this setting. If the structured model improves once I actually train it correctly, then the conclusion will be that these labelled embeddings do improve performance.

- What roadblocks have you hit so far?

Prompting the agent is a challenge for me, I don't have much experience using them, and it does not always listen to me. It is also difficult for me to interpret the massive amounts of code it puts out in a timely manner, and as someone who really thinks paying attention to detail is valuable, this is frustrating. Ensuring everything is correct is important for a project like this.

I still need to rerun the structured experiment with the updated full fine-tuning setup so the comparison reflects the corrected training procedure.

