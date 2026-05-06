"""Generate synthetic algebra datasets for baseline and structured training."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable

import sympy as sp


x = sp.Symbol("x")


@dataclass
class Example:
    prompt: str
    output: str

    def to_dict(self) -> dict:
        return {
            "prompt": self.prompt,
            "output": self.output,
        }


class AlgebraDatasetGenerator:
    """Create synthetic algebra prompt/output pairs with SymPy-checked targets."""

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def rand_int(self, low: int = -5, high: int = 5, exclude_zero: bool = False) -> int:
        while True:
            n = self.rng.randint(low, high)
            if exclude_zero and n == 0:
                continue
            return n

    def rand_linear_expr(self):
        a = self.rand_int(-5, 5, exclude_zero=True)
        b = self.rand_int(-5, 5)
        return sp.Integer(a) * x + sp.Integer(b)

    def rand_binomial(self):
        a = self.rand_int(1, 5)
        b = self.rand_int(-5, 5)
        return sp.Integer(a) * (x + sp.Integer(b))

    def rand_product_of_binomials(self):
        a = self.rand_int(-5, 5, exclude_zero=True)
        b = self.rand_int(-5, 5)
        c = self.rand_int(-5, 5)
        return (x + sp.Integer(b)) * (sp.Integer(a) * x + sp.Integer(c))

    def rand_factorable_quadratic(self):
        r1 = self.rand_int(-5, 5)
        r2 = self.rand_int(-5, 5)
        return sp.expand((x - sp.Integer(r1)) * (x - sp.Integer(r2)))

    def rand_repeated_like_terms(self):
        a = self.rand_int(1, 5)
        b = self.rand_int(1, 5)
        c = self.rand_int(-5, 5)
        return sp.Integer(a) * x + sp.Integer(b) * x + sp.Integer(c)

    def make_simplify_example(self) -> Example:
        expr = self.rand_repeated_like_terms()
        target = sp.simplify(expr)
        return Example(
            prompt=f"simplify {sp.sstr(expr)} =>",
            output=sp.sstr(target),
        )

    def make_expand_example(self) -> Example:
        expr = self.rand_binomial() if self.rng.random() < 0.5 else self.rand_product_of_binomials()
        target = sp.expand(expr)
        return Example(
            prompt=f"expand {sp.sstr(expr)} =>",
            output=sp.sstr(target),
        )

    def make_factor_example(self) -> Example:
        expr = self.rand_factorable_quadratic()
        target = sp.factor(expr)
        return Example(
            prompt=f"factor {sp.sstr(expr)} =>",
            output=sp.sstr(target),
        )

    def make_substitute_example(self) -> Example:
        expr = self.rand_product_of_binomials() if self.rng.random() < 0.5 else self.rand_linear_expr()
        value = self.rand_int(-3, 3)
        target = sp.simplify(expr.subs(x, sp.Integer(value)))
        return Example(
            prompt=f"substitute x = {value} into {sp.sstr(expr)} =>",
            output=sp.sstr(target),
        )

    def make_solve_example(self) -> Example:
        a = self.rand_int(-5, 5, exclude_zero=True)
        sol = self.rand_int(-5, 5)
        b = self.rand_int(-5, 5)
        c = a * sol + b
        lhs = sp.Integer(a) * x + sp.Integer(b)
        rhs = sp.Integer(c)
        return Example(
            prompt=f"solve {sp.sstr(lhs)} = {sp.sstr(rhs)} for x =>",
            output=f"x = {sol}",
        )

    def sample_example(self, task: str | None = None) -> Example:
        builders: dict[str, Callable[[], Example]] = {
            "simplify": self.make_simplify_example,
            "expand": self.make_expand_example,
            "factor": self.make_factor_example,
            "substitute": self.make_substitute_example,
            "solve": self.make_solve_example,
        }
        if task is None:
            task = self.rng.choice(list(builders.keys()))
        if task not in builders:
            raise ValueError(f"Unknown task: {task}")
        return builders[task]()

    def generate(self, n: int, task_weights: dict[str, float] | None = None) -> list[Example]:
        """Sample a mixed task dataset, optionally with custom per-task weights."""

        tasks = ["simplify", "expand", "factor", "substitute", "solve"]
        weights = [1.0, 1.0, 1.0, 1.0, 1.0]

        if task_weights is not None:
            weights = [float(task_weights.get(task, 0.0)) for task in tasks]
            if sum(weights) <= 0:
                raise ValueError("task_weights must contain at least one positive weight")

        examples: list[Example] = []
        for _ in range(n):
            task = self.rng.choices(tasks, weights=weights, k=1)[0]
            examples.append(self.sample_example(task))
        return examples


def train_val_test_split(
    items: list[Example],
    train_frac: float = 0.85,
    val_frac: float = 0.08,
) -> tuple[list[Example], list[Example], list[Example]]:
    """Split a shuffled example list into train/validation/test slices."""

    n = len(items)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train = items[:n_train]
    val = items[n_train:n_train + n_val]
    test = items[n_train + n_val:]
    return train, val, test


def save_jsonl(path: Path, examples: list[Example]) -> None:
    """Write examples in the JSONL format consumed by the trainers."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex.to_dict(), ensure_ascii=False) + "\n")


def main() -> None:
    out_dir = Path("data")
    gen = AlgebraDatasetGenerator(seed=42)

    examples = gen.generate(6000)
    gen.rng.shuffle(examples)

    train, val, test = train_val_test_split(examples)

    save_jsonl(out_dir / "train.jsonl", train)
    save_jsonl(out_dir / "val.jsonl", val)
    save_jsonl(out_dir / "test.jsonl", test)

    print(f"Saved {len(train)} train examples")
    print(f"Saved {len(val)} val examples")
    print(f"Saved {len(test)} test examples")
    print("Sample example:")
    print(json.dumps(train[0].to_dict(), indent=2))


if __name__ == "__main__":
    main()
