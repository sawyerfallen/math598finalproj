"""Node-type vocabulary used by the structured GPT-2 small baseline."""

from __future__ import annotations


NODE_TYPE_TO_ID = {
    # Task words such as "simplify" or "solve".
    "TASK": 0,
    # Symbolic variables like x, y, or z.
    "VARIABLE": 1,
    # Integer literals, including signed constants when parsed as one token.
    "CONSTANT": 2,
    # Additive operators + and - when used as operators.
    "ADD": 3,
    # Multiplication operator *.
    "MUL": 4,
    # Exponent operators ^ and **.
    "POW": 5,
    # Equality sign =.
    "EQUALITY": 6,
    # Structural punctuation such as parentheses and =>.
    "PUNCT": 7,
    # Fallback bucket for anything not covered above.
    "OTHER": 8,
}

ID_TO_NODE_TYPE = {node_id: node_type for node_type, node_id in NODE_TYPE_TO_ID.items()}
NUM_NODE_TYPES = len(NODE_TYPE_TO_ID)
