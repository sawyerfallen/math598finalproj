# Solve-Only Language Specification

The final experiment uses controlled symbolic equations over one variable, `x`.

## Prompts

Every prompt has this form:

```text
solve <equation> for x =>
```

Examples:

```text
solve 2*x + 3 = 7 for x =>
solve 3*(x - 1) - 5 = -17 for x =>
solve -x - 6 = x - 8 for x =>
solve x**2 - 2*x - 3 = 0 for x =>
```

## Outputs

Linear equations use one canonical integer solution:

```text
x = 2
```

Quadratics with two real integer roots use sorted roots:

```text
x = -1 or x = 3
```

## Node Types

The structured model labels prompt tokens with coarse symbolic types:

- `TASK` for words such as `solve` and `for`
- `VARIABLE` for `x`
- `CONSTANT` for integer literals
- `ADD`, `MUL`, and `POW` for operators
- `EQUALITY` for `=`
- `PUNCT` for parentheses and `=>`
- `OTHER` for tokenizer pieces that do not align cleanly to one symbolic token
