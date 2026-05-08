"""Generate synthetic algebra datasets for baseline and structured training."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import sympy as sp


x = sp.Symbol("x")
TASKS = ("simplify", "expand", "factor", "substitute", "solve")
SOLVE_DIFFICULTIES = ("easy", "hard", "mixed")


@dataclass
class Example:
    """One clean training example plus optional sidecar evaluation metadata."""

    prompt: str
    output: str
    task: str
    difficulty: str | None = None
    solve_kind: str | None = None

    def to_dict(self) -> dict:
        return {
            "prompt": self.prompt,
            "output": self.output,
        }

    def metadata_dict(self, sample_index: int) -> dict:
        """Return metadata for evaluation without adding leakage to training JSONL."""

        return {
            "sample_index": sample_index,
            "prompt": self.prompt,
            "output": self.output,
            "task": self.task,
            "difficulty": self.difficulty,
            "solve_kind": self.solve_kind,
        }


class AlgebraDatasetGenerator:
    """Create synthetic algebra prompt/output pairs with SymPy-checked targets."""

    def __init__(self, seed: int = 0, solve_difficulty: str = "easy"):
        self.rng = random.Random(seed)
        if solve_difficulty not in SOLVE_DIFFICULTIES:
            raise ValueError(f"Unknown solve difficulty: {solve_difficulty}")
        self.solve_difficulty = solve_difficulty

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
            task="simplify",
        )

    def make_expand_example(self) -> Example:
        expr = self.rand_binomial() if self.rng.random() < 0.5 else self.rand_product_of_binomials()
        target = sp.expand(expr)
        return Example(
            prompt=f"expand {sp.sstr(expr)} =>",
            output=sp.sstr(target),
            task="expand",
        )

    def make_factor_example(self) -> Example:
        expr = self.rand_factorable_quadratic()
        target = sp.factor(expr)
        return Example(
            prompt=f"factor {sp.sstr(expr)} =>",
            output=sp.sstr(target),
            task="factor",
        )

    def make_substitute_example(self) -> Example:
        expr = self.rand_product_of_binomials() if self.rng.random() < 0.5 else self.rand_linear_expr()
        value = self.rand_int(-3, 3)
        target = sp.simplify(expr.subs(x, sp.Integer(value)))
        return Example(
            prompt=f"substitute x = {value} into {sp.sstr(expr)} =>",
            output=sp.sstr(target),
            task="substitute",
        )

    @staticmethod
    def format_solutions(solutions: list[int]) -> str:
        """Format solve outputs canonically, including two-root answers."""

        unique_solutions = sorted(set(int(solution) for solution in solutions))
        if len(unique_solutions) == 1:
            return f"x = {unique_solutions[0]}"
        return " or ".join(f"x = {solution}" for solution in unique_solutions)

    @staticmethod
    def format_x_term(coefficient: int) -> str:
        if coefficient == 1:
            return "x"
        if coefficient == -1:
            return "-x"
        return f"{coefficient}*x"

    @staticmethod
    def join_terms(terms: list[str]) -> str:
        """Join signed term strings into a readable algebra expression."""

        cleaned_terms = [term for term in terms if term and term != "0"]
        if not cleaned_terms:
            return "0"

        expression = ""
        for term in cleaned_terms:
            if term.startswith("-"):
                separator = " - " if expression else "-"
                expression += separator + term[1:]
            else:
                separator = " + " if expression else ""
                expression += separator + term
        return expression

    @staticmethod
    def format_x_shift(shift: int) -> str:
        return AlgebraDatasetGenerator.join_terms(["x", str(shift)])

    @staticmethod
    def format_scaled_parenthesized(coefficient: int, shift: int) -> str:
        return f"{coefficient}*({AlgebraDatasetGenerator.format_x_shift(shift)})"

    def make_easy_solve_example(self) -> Example:
        a = self.rand_int(-5, 5, exclude_zero=True)
        sol = self.rand_int(-5, 5)
        b = self.rand_int(-5, 5)
        c = a * sol + b
        lhs = sp.Integer(a) * x + sp.Integer(b)
        rhs = sp.Integer(c)
        return Example(
            prompt=f"solve {sp.sstr(lhs)} = {sp.sstr(rhs)} for x =>",
            output=self.format_solutions([sol]),
            task="solve",
            difficulty="easy",
            solve_kind="linear_easy",
        )

    def make_parenthesized_linear_solve_example(self) -> Example:
        """Solve a linear equation with distribution on one side."""

        a = self.rand_int(-5, 5, exclude_zero=True)
        b = self.rand_int(-5, 5)
        c = self.rand_int(-5, 5)
        sol = self.rand_int(-5, 5)
        lhs = sp.Integer(a) * (x + sp.Integer(b)) + sp.Integer(c)
        rhs = sp.simplify(lhs.subs(x, sol))
        lhs_text = self.join_terms([self.format_scaled_parenthesized(a, b), str(c)])
        return Example(
            prompt=f"solve {lhs_text} = {sp.sstr(rhs)} for x =>",
            output=self.format_solutions([sol]),
            task="solve",
            difficulty="hard",
            solve_kind="parenthesized_linear",
        )

    def make_both_sides_linear_solve_example(self) -> Example:
        """Solve a linear equation with x terms on both sides."""

        lhs_coeff = self.rand_int(-6, 6, exclude_zero=True)
        rhs_coeff = self.rand_int(-6, 6, exclude_zero=True)
        while lhs_coeff == rhs_coeff:
            rhs_coeff = self.rand_int(-6, 6, exclude_zero=True)

        lhs_const = self.rand_int(-10, 10)
        sol = self.rand_int(-6, 6)
        rhs_const = (lhs_coeff - rhs_coeff) * sol + lhs_const
        lhs = sp.Integer(lhs_coeff) * x + sp.Integer(lhs_const)
        rhs = sp.Integer(rhs_coeff) * x + sp.Integer(rhs_const)
        return Example(
            prompt=f"solve {sp.sstr(lhs)} = {sp.sstr(rhs)} for x =>",
            output=self.format_solutions([sol]),
            task="solve",
            difficulty="hard",
            solve_kind="x_both_sides",
        )

    def make_collect_like_terms_solve_example(self) -> Example:
        """Solve after collecting repeated x terms."""

        a = self.rand_int(-5, 5, exclude_zero=True)
        b = self.rand_int(-5, 5, exclude_zero=True)
        c = self.rand_int(-5, 5)
        d = self.rand_int(-5, 5)
        while a + b == d:
            d = self.rand_int(-5, 5)
        sol = self.rand_int(-6, 6)
        rhs_const = (a + b - d) * sol + c
        lhs_text = self.join_terms([self.format_x_term(a), self.format_x_term(b), str(c)])
        rhs_text = self.join_terms([self.format_x_term(d), str(rhs_const)])
        return Example(
            prompt=f"solve {lhs_text} = {rhs_text} for x =>",
            output=self.format_solutions([sol]),
            task="solve",
            difficulty="hard",
            solve_kind="collect_like_terms",
        )

    def make_distribution_both_sides_solve_example(self) -> Example:
        """Solve a linear equation that requires distribution on both sides."""

        lhs_coeff = self.rand_int(-5, 5, exclude_zero=True)
        rhs_coeff = self.rand_int(-5, 5, exclude_zero=True)
        while lhs_coeff == rhs_coeff:
            rhs_coeff = self.rand_int(-5, 5, exclude_zero=True)

        lhs_shift = self.rand_int(-5, 5)
        rhs_shift = self.rand_int(-5, 5)
        lhs_const = self.rand_int(-5, 5)
        sol = self.rand_int(-6, 6)
        lhs = sp.Integer(lhs_coeff) * (x + sp.Integer(lhs_shift)) + sp.Integer(lhs_const)
        rhs_without_const = sp.Integer(rhs_coeff) * (x + sp.Integer(rhs_shift))
        rhs_const = sp.simplify(lhs.subs(x, sol) - rhs_without_const.subs(x, sol))
        lhs_text = self.join_terms([self.format_scaled_parenthesized(lhs_coeff, lhs_shift), str(lhs_const)])
        rhs_text = self.join_terms(
            [self.format_scaled_parenthesized(rhs_coeff, rhs_shift), str(int(rhs_const))]
        )
        return Example(
            prompt=f"solve {lhs_text} = {rhs_text} for x =>",
            output=self.format_solutions([sol]),
            task="solve",
            difficulty="hard",
            solve_kind="distribution_both_sides",
        )

    def make_quadratic_solve_example(self) -> Example:
        """Solve a simple factorable quadratic with two real integer roots."""

        root_a = self.rand_int(-6, 6)
        root_b = self.rand_int(-6, 6)
        while root_b == root_a:
            root_b = self.rand_int(-6, 6)
        leading_coeff = self.rand_int(1, 3)
        lhs = sp.expand(sp.Integer(leading_coeff) * (x - sp.Integer(root_a)) * (x - sp.Integer(root_b)))
        rhs = sp.Integer(0)
        return Example(
            prompt=f"solve {sp.sstr(lhs)} = {sp.sstr(rhs)} for x =>",
            output=self.format_solutions([root_a, root_b]),
            task="solve",
            difficulty="hard",
            solve_kind="quadratic_two_real_roots",
        )

    def make_hard_solve_example(self) -> Example:
        """Sample controlled harder solve-for-x equations."""

        builders = [
            self.make_parenthesized_linear_solve_example,
            self.make_both_sides_linear_solve_example,
            self.make_collect_like_terms_solve_example,
            self.make_distribution_both_sides_solve_example,
            self.make_quadratic_solve_example,
        ]
        return self.rng.choice(builders)()

    def make_solve_example(self) -> Example:
        if self.solve_difficulty == "easy":
            return self.make_easy_solve_example()
        if self.solve_difficulty == "hard":
            return self.make_hard_solve_example()
        return self.make_easy_solve_example() if self.rng.random() < 0.35 else self.make_hard_solve_example()

    def builders(self) -> dict[str, Callable[[], Example]]:
        """Return the available task-specific example builders."""

        return {
            "simplify": self.make_simplify_example,
            "expand": self.make_expand_example,
            "factor": self.make_factor_example,
            "substitute": self.make_substitute_example,
            "solve": self.make_solve_example,
        }

    def sample_example(self, task: str | None = None) -> Example:
        builders = self.builders()
        if task is None:
            task = self.rng.choice(list(builders.keys()))
        if task not in builders:
            raise ValueError(f"Unknown task: {task}")
        return builders[task]()

    def generate(
        self,
        n: int,
        tasks: list[str] | None = None,
        task_weights: dict[str, float] | None = None,
    ) -> list[Example]:
        """Sample task examples, optionally limiting generation to a task subset."""

        available_builders = self.builders()
        selected_tasks = tasks or list(TASKS)
        unknown_tasks = sorted(set(selected_tasks) - set(available_builders))
        if unknown_tasks:
            raise ValueError(f"Unknown task(s): {', '.join(unknown_tasks)}")

        weights = [1.0 for _ in selected_tasks]

        if task_weights is not None:
            weights = [float(task_weights.get(task, 0.0)) for task in selected_tasks]
            if sum(weights) <= 0:
                raise ValueError("task_weights must contain at least one positive weight")

        examples: list[Example] = []
        for _ in range(n):
            task = self.rng.choices(selected_tasks, weights=weights, k=1)[0]
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


def split_by_counts(
    items: list[Example],
    train_size: int,
    val_size: int,
    test_size: int,
) -> tuple[list[Example], list[Example], list[Example]]:
    """Split a shuffled example list by exact requested counts."""

    expected_total = train_size + val_size + test_size
    if len(items) != expected_total:
        raise ValueError(f"Expected {expected_total} examples for explicit split sizes, got {len(items)}")
    train = items[:train_size]
    val = items[train_size:train_size + val_size]
    test = items[train_size + val_size:]
    return train, val, test


def example_key(example: Example) -> tuple[str, str]:
    """Use the actual model-visible example as the deduplication identity."""

    return example.prompt, example.output


def deduplicate_examples(examples: list[Example]) -> list[Example]:
    """Keep the first occurrence of each prompt/output pair."""

    seen: set[tuple[str, str]] = set()
    unique_examples: list[Example] = []
    for example in examples:
        key = example_key(example)
        if key in seen:
            continue
        seen.add(key)
        unique_examples.append(example)
    return unique_examples


def generate_deduplicated(
    generator: AlgebraDatasetGenerator,
    target_unique_count: int,
    tasks: list[str],
    max_raw_examples: int,
    batch_size: int,
) -> tuple[list[Example], int]:
    """Generate examples until enough unique prompt/output pairs are available."""

    if batch_size <= 0:
        raise ValueError("--dedup-batch-size must be positive")
    if max_raw_examples < target_unique_count:
        raise ValueError("--max-raw-examples must be at least the requested unique dataset size")

    seen: set[tuple[str, str]] = set()
    unique_examples: list[Example] = []
    raw_count = 0

    while len(unique_examples) < target_unique_count and raw_count < max_raw_examples:
        remaining_raw_budget = max_raw_examples - raw_count
        sample_count = min(batch_size, remaining_raw_budget)
        for example in generator.generate(sample_count, tasks=tasks):
            raw_count += 1
            key = example_key(example)
            if key in seen:
                continue
            seen.add(key)
            unique_examples.append(example)
            if len(unique_examples) >= target_unique_count:
                break

    return unique_examples, raw_count


def scaled_split_by_requested_counts(
    items: list[Example],
    requested_train_size: int,
    requested_val_size: int,
    requested_test_size: int,
) -> tuple[list[Example], list[Example], list[Example]]:
    """Scale requested split counts down if deduplication cannot fill them."""

    requested_total = requested_train_size + requested_val_size + requested_test_size
    if len(items) >= requested_total:
        return split_by_counts(items[:requested_total], requested_train_size, requested_val_size, requested_test_size)

    train_size = int(len(items) * requested_train_size / requested_total)
    val_size = int(len(items) * requested_val_size / requested_total)
    test_size = len(items) - train_size - val_size
    return split_by_counts(items, train_size, val_size, test_size)


def cross_split_duplicate_count(*splits: list[Example]) -> int:
    """Count examples that appear in more than one split."""

    split_memberships: dict[tuple[str, str], set[int]] = {}
    for split_index, split in enumerate(splits):
        for example in split:
            split_memberships.setdefault(example_key(example), set()).add(split_index)
    return sum(1 for memberships in split_memberships.values() if len(memberships) > 1)


def count_by_field(examples: list[Example], field_name: str) -> dict[str, int]:
    """Count metadata values such as difficulty or solve_kind."""

    return dict(sorted(Counter(str(getattr(ex, field_name) or "none") for ex in examples).items()))


def save_jsonl(path: Path, examples: list[Example]) -> None:
    """Write examples in the JSONL format consumed by the trainers."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex.to_dict(), ensure_ascii=False) + "\n")


def save_metadata_jsonl(path: Path, examples: list[Example]) -> None:
    """Write sidecar metadata used for grouped evaluation and plotting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for sample_index, ex in enumerate(examples):
            f.write(json.dumps(ex.metadata_dict(sample_index), ensure_ascii=False) + "\n")


def write_generation_summary(path: Path, summary: dict) -> None:
    """Save machine-readable dataset generation and deduplication facts."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def write_generation_summary_text(path: Path, summary: dict) -> None:
    """Save a short human-readable companion to the JSON summary."""

    lines = [
        "Dataset Generation Summary",
        f"Output directory: {summary['output_dir']}",
        f"Tasks: {', '.join(summary['tasks'])}",
        f"Solve difficulty: {summary['solve_difficulty']}",
        f"Seed: {summary['seed']}",
        f"Deduplicated before split: {summary['deduplicated_before_split']}",
        f"Raw generated examples: {summary['raw_generated_examples']}",
        f"Unique examples after deduplication: {summary['unique_examples_after_deduplication']}",
        f"Duplicate examples removed: {summary['duplicate_examples_removed']}",
        f"Requested split sizes: {summary['requested_split_sizes']}",
        f"Final split sizes: {summary['final_split_sizes']}",
        f"Exact requested split sizes achieved: {summary['exact_requested_split_sizes_achieved']}",
        f"Cross-split duplicate count: {summary['cross_split_duplicate_count']}",
        f"Difficulty counts: {summary['difficulty_counts']}",
        f"Solve kind counts: {summary['solve_kind_counts']}",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate train/val/test JSONL splits for symbolic algebra.")
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--dataset-size", type=int, default=6000)
    parser.add_argument("--train-size", type=int, default=None)
    parser.add_argument("--val-size", type=int, default=None)
    parser.add_argument("--test-size", type=int, default=None)
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=TASKS,
        default=list(TASKS),
        help="Task subset to generate, for example: --tasks solve",
    )
    parser.add_argument(
        "--solve-difficulty",
        choices=SOLVE_DIFFICULTIES,
        default="easy",
        help="Difficulty distribution used when generating solve examples.",
    )
    parser.add_argument("--train-frac", type=float, default=0.85)
    parser.add_argument("--val-frac", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--deduplicate-before-split",
        action="store_true",
        help="Deduplicate by prompt/output before train/val/test splitting.",
    )
    parser.add_argument(
        "--max-raw-examples",
        type=int,
        default=100000,
        help="Maximum raw samples to draw when --deduplicate-before-split is enabled.",
    )
    parser.add_argument(
        "--dedup-batch-size",
        type=int,
        default=5000,
        help="Raw samples to draw per loop while collecting unique examples.",
    )
    parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="Do not write split_metadata.jsonl sidecars for evaluation grouping.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir
    gen = AlgebraDatasetGenerator(seed=args.seed, solve_difficulty=args.solve_difficulty)
    requested_split_sizes = {
        "train": args.train_size,
        "val": args.val_size,
        "test": args.test_size,
    }

    explicit_split_sizes = [args.train_size, args.val_size, args.test_size]
    if any(size is not None for size in explicit_split_sizes):
        if not all(size is not None for size in explicit_split_sizes):
            raise ValueError("--train-size, --val-size, and --test-size must be provided together")
        args.dataset_size = int(args.train_size + args.val_size + args.test_size)

    if args.deduplicate_before_split:
        examples, raw_generated_examples = generate_deduplicated(
            generator=gen,
            target_unique_count=args.dataset_size,
            tasks=args.tasks,
            max_raw_examples=args.max_raw_examples,
            batch_size=args.dedup_batch_size,
        )
    else:
        examples = gen.generate(args.dataset_size, tasks=args.tasks)
        raw_generated_examples = len(examples)

    unique_examples = deduplicate_examples(examples)
    if args.deduplicate_before_split:
        examples = unique_examples
    gen.rng.shuffle(examples)

    if all(size is not None for size in explicit_split_sizes):
        train, val, test = scaled_split_by_requested_counts(examples, args.train_size, args.val_size, args.test_size)
    else:
        train, val, test = train_val_test_split(
            examples,
            train_frac=args.train_frac,
            val_frac=args.val_frac,
        )

    save_jsonl(out_dir / "train.jsonl", train)
    save_jsonl(out_dir / "val.jsonl", val)
    save_jsonl(out_dir / "test.jsonl", test)
    if not args.no_metadata:
        save_metadata_jsonl(out_dir / "train_metadata.jsonl", train)
        save_metadata_jsonl(out_dir / "val_metadata.jsonl", val)
        save_metadata_jsonl(out_dir / "test_metadata.jsonl", test)

    requested_total = (
        args.train_size + args.val_size + args.test_size
        if all(size is not None for size in explicit_split_sizes)
        else args.dataset_size
    )
    final_split_sizes = {"train": len(train), "val": len(val), "test": len(test)}
    summary = {
        "output_dir": str(out_dir),
        "tasks": args.tasks,
        "solve_difficulty": args.solve_difficulty,
        "seed": args.seed,
        "deduplicated_before_split": args.deduplicate_before_split,
        "deduplication_identity": "prompt/output",
        "raw_generated_examples": raw_generated_examples,
        "unique_examples_after_deduplication": len(unique_examples),
        "duplicate_examples_removed": raw_generated_examples - len(unique_examples),
        "requested_dataset_size": requested_total,
        "requested_split_sizes": requested_split_sizes,
        "final_split_sizes": final_split_sizes,
        "exact_requested_split_sizes_achieved": sum(final_split_sizes.values()) == requested_total,
        "cross_split_duplicate_count": cross_split_duplicate_count(train, val, test),
        "difficulty_counts": count_by_field(examples, "difficulty"),
        "solve_kind_counts": count_by_field(examples, "solve_kind"),
        "split_difficulty_counts": {
            "train": count_by_field(train, "difficulty"),
            "val": count_by_field(val, "difficulty"),
            "test": count_by_field(test, "difficulty"),
        },
        "split_solve_kind_counts": {
            "train": count_by_field(train, "solve_kind"),
            "val": count_by_field(val, "solve_kind"),
            "test": count_by_field(test, "solve_kind"),
        },
        "max_raw_examples": args.max_raw_examples,
        "dedup_batch_size": args.dedup_batch_size,
    }
    write_generation_summary(out_dir / "generation_summary.json", summary)
    write_generation_summary_text(out_dir / "generation_summary.txt", summary)

    print(f"Dataset directory: {out_dir}")
    print(f"Tasks: {', '.join(args.tasks)}")
    print(f"Solve difficulty: {args.solve_difficulty}")
    print(f"Deduplicated before split: {args.deduplicate_before_split}")
    print(f"Raw generated examples: {raw_generated_examples}")
    print(f"Unique examples after deduplication: {len(unique_examples)}")
    print(f"Duplicate examples removed: {raw_generated_examples - len(unique_examples)}")
    print(f"Total examples used for splitting: {len(examples)}")
    print(f"Saved {len(train)} train examples")
    print(f"Saved {len(val)} val examples")
    print(f"Saved {len(test)} test examples")
    print(f"Cross-split duplicate count: {summary['cross_split_duplicate_count']}")
    if not args.no_metadata:
        print(f"Saved metadata sidecars to {out_dir}")
        print(f"Difficulty counts: {summary['difficulty_counts']}")
        print(f"Solve kind counts: {summary['solve_kind_counts']}")
    print(f"Saved generation summary to {out_dir / 'generation_summary.json'}")
    print("Sample example:")
    print(json.dumps(train[0].to_dict(), indent=2))


if __name__ == "__main__":
    main()
