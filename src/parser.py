"""Simple prompt parser that emits symbolic tokens and node-type labels."""

from __future__ import annotations


TASK_WORDS = {
    "solve",
    "for",
}
VARIABLE_NAMES = {"x", "y", "z"}
# Tokens after which a leading + or - should be treated as part of a number.
UNARY_PREFIX_TOKENS = {"(", "=", "+", "-", "*", "**", "^", "=>"}


def _consume_word(text: str, start: int) -> tuple[str, int]:
    """Read one contiguous alphabetic/identifier token."""

    end = start
    while end < len(text) and (text[end].isalnum() or text[end] == "_"):
        end += 1
    return text[start:end], end


def _consume_number(text: str, start: int) -> tuple[str, int]:
    """Read one integer token, optionally with a leading sign."""

    end = start
    if text[end] in "+-":
        end += 1
    while end < len(text) and text[end].isdigit():
        end += 1
    return text[start:end], end


def _allows_signed_number(previous_token: str | None) -> bool:
    """Return True when +/- should start a signed constant instead of an operator."""

    return previous_token is None or previous_token in UNARY_PREFIX_TOKENS


def _classify_word(token: str) -> str:
    """Map identifier-like tokens to their coarse node type."""

    if token in VARIABLE_NAMES:
        return "VARIABLE"
    if token.lower() in TASK_WORDS:
        return "TASK"
    return "OTHER"


def tokenize_prompt_with_node_types(prompt: str) -> tuple[list[str], list[str]]:
    """Tokenize a prompt into symbolic tokens and coarse node-type labels."""

    tokens: list[str] = []
    node_types: list[str] = []
    index = 0
    previous_token: str | None = None

    while index < len(prompt):
        char = prompt[index]

        if char.isspace():
            # Whitespace separates symbolic tokens but is not itself labeled.
            index += 1
            continue

        if prompt.startswith("=>", index):
            # Treat the answer arrow as a single punctuation token.
            tokens.append("=>")
            node_types.append("PUNCT")
            previous_token = "=>"
            index += 2
            continue

        if prompt.startswith("**", index):
            # SymPy-style exponent notation appears in the dataset.
            tokens.append("**")
            node_types.append("POW")
            previous_token = "**"
            index += 2
            continue

        if char == "^":
            # Also accept caret notation for exponentiation.
            tokens.append("^")
            node_types.append("POW")
            previous_token = "^"
            index += 1
            continue

        if char in "()":
            # Parentheses are structural punctuation.
            tokens.append(char)
            node_types.append("PUNCT")
            previous_token = char
            index += 1
            continue

        if char == "=":
            # Equality is its own node type because solve prompts use it semantically.
            tokens.append("=")
            node_types.append("EQUALITY")
            previous_token = char
            index += 1
            continue

        if char == "*":
            # Multiplication is always labeled directly.
            tokens.append("*")
            node_types.append("MUL")
            previous_token = char
            index += 1
            continue

        if char in "+-":
            next_is_digit = index + 1 < len(prompt) and prompt[index + 1].isdigit()
            if next_is_digit and _allows_signed_number(previous_token):
                # Parse signed literals like -3 as one CONSTANT token when the sign is unary.
                token, index = _consume_number(prompt, index)
                tokens.append(token)
                node_types.append("CONSTANT")
                previous_token = token
            else:
                # Otherwise +/- acts as an additive operator.
                tokens.append(char)
                node_types.append("ADD")
                previous_token = char
                index += 1
            continue

        if char.isdigit():
            # Unsigned integer constant.
            token, index = _consume_number(prompt, index)
            tokens.append(token)
            node_types.append("CONSTANT")
            previous_token = token
            continue

        if char.isalpha() or char == "_":
            # Task words and variables both come through this path.
            token, index = _consume_word(prompt, index)
            tokens.append(token)
            node_types.append(_classify_word(token))
            previous_token = token
            continue

        # Keep a safe fallback for punctuation or symbols outside the minimal grammar.
        tokens.append(char)
        node_types.append("OTHER")
        previous_token = char
        index += 1

    return tokens, node_types
