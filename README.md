Project Plan:

1. For the purposes of this experiment/project, I will be creating a pre-defined symbolic algebra notation so I can define the exact types of equations I want to predict. For example:

> expand: 2*(x+3) = 2*x+6

> factor: x^2-1 = (x-1)*(x+1)

> simplify: x+x+2 = 2*x+2

> substitute: f(x)=x^2+1; f(2) = 5

> solve: 2*x+3=7; x = 2

2. Create/find a parser so I can convert these into labels to use for the encoding. 

3. Generate a dataset, maybe using some other library to SymPy to evaluate their validity. 

4. Train/fine tune a model like Pythia 70M on these in order to get the baseline prediction accuracy

5. Implement the encodings and run the experiment.

6. Analyze results