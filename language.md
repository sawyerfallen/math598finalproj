Language Specification:

- Variables:
    - x, y, z
- Numbers:
    - Going to restrict to just a few integers for the purposes of this experiment, like -5 to 5.
- Operators:
    - +, -, *, /, ^
    - Parentheses too
- Equals sign


Tasks for LLM:

Here are some examples of basic algebra tasks for the LLM to solve, with the prompt and the correct output:

Prompt: simplify x + x + 2
Output: 2*x + 2

Prompt: expand 2*(x + 3)
Output: 2*x + 6

Prompt: expand (x + 1)*(x + 2)
Output: x^2 + 3*x + 2

Prompt: factor x^2 - 1
Output: (x - 1)*(x + 1)

Prompt: factor x^2 + 3*x + 2
Output: (x + 1)*(x + 2)

Prompt: substitute x = 3 into x^2 + 1
Output: 10

Prompt: substitute x = 4 into 2*x + 5
Output: 13

Prompt: solve 2*x + 3 = 7 for x
Output: x = 2

Prompt: solve x - 4 = 1 for x
Output: x = 5

Prompt: solve 3*x = 12 for x
Output: x = 4