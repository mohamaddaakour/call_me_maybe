"""JSON scalar grammars that keep every generated value parseable."""

from __future__ import annotations

import re
from typing import Any

from src.models import JsonType

# regex to check integers
_INTEGER = re.compile(r"-?(0|[1-9][0-9]*)")
_INTEGER_PREFIX = re.compile(r"-?(0|[1-9][0-9]*)?")

# regex to check any number even double numbers
_NUMBER = re.compile(r"-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?")
_NUMBER_PREFIX = re.compile(
    r"-?"
    r"|-?(0|[1-9][0-9]*)"
    r"|-?(0|[1-9][0-9]*)\.[0-9]*"
    r"|-?(0|[1-9][0-9]*)(\.[0-9]+)?[eE][+-]?[0-9]*"
)

_BOOLEANS = ("true", "false")

_ESCAPES = '"\\/bfnrt'
_HEX_DIGITS = "0123456789abcdefABCDEF"


# It returns two True/False values:
# 1. Can the text still become a valid JSON string?
# 2. Is the text already a complete JSON string?
def _scan_string(text: str) -> tuple[bool, bool]:
    """Walk a JSON string once.

    Args:
        text: The value decoded so far, opening quote included.

    Returns:
        Two flags: whether the text can still grow into a JSON string,
        and whether it already is a complete one.
    """
    if not text.startswith('"'):
        return (text == "", False)

    index = 1

    while index < len(text):
        character = text[index]

        if character == '"':
            finished = index == len(text) - 1
            return (finished, finished)

        if character == "\\":
            escape = text[index + 1:index + 2]

            if escape == "":
                return (True, False)

            if escape in _ESCAPES:
                index += 2
                continue

            if escape == "u":
                digits = text[index + 2:index + 6]

                if any(digit not in _HEX_DIGITS for digit in digits):
                    return (False, False)

                if len(digits) < 4:
                    return (True, False)

                index += 6
                continue

            return (False, False)

        if character < " ":
            return (False, False)

        index += 1

    return (True, False)


def is_prefix(text: str, json_type: JsonType) -> bool:
    """Report whether text can still grow into a value of this type."""
    if json_type == "string":
        return _scan_string(text)[0]

    if json_type == "boolean":
        return any(word.startswith(text) for word in _BOOLEANS)

    if json_type == "integer":
        return _INTEGER_PREFIX.fullmatch(text) is not None

    return _NUMBER_PREFIX.fullmatch(text) is not None


def is_complete(text: str, json_type: JsonType) -> bool:
    """Report whether text is already a finished value of this type."""
    if json_type == "string":
        return _scan_string(text)[1]

    if json_type == "boolean":
        return text in _BOOLEANS

    if json_type == "integer":
        return _INTEGER.fullmatch(text) is not None

    return _NUMBER.fullmatch(text) is not None


def is_closed(text: str, json_type: JsonType) -> bool:
    """Report whether a finished value can no longer be extended.

    A closed value ends the decoding loop without spending a forward pass
    on a token that the grammar would reject anyway. A complete string or
    boolean is closed: nothing may follow the final quote, and neither
    "true" nor "false" is the prefix of a longer word. A number never is,
    because "1" can still become "1.5".
    """
    if json_type in ("integer", "number"):
        return False

    return is_complete(text, json_type)


def prefill(json_type: JsonType) -> str:
    """Give the model the part of the value that is structure, not content.

    The opening quote of a string carries no meaning, so writing it for the
    model removes the one place it likes to answer with prose instead.
    """
    return '"' if json_type == "string" else ""


def placeholder(json_type: JsonType) -> Any:
    """Give the empty value of a type, used when decoding cannot finish.

    A prompt the model cannot answer still owes the output file one
    schema-valid object, so the value is empty rather than absent.
    """
    if json_type == "string":
        return ""

    if json_type == "boolean":
        return False

    if json_type == "integer":
        return 0

    return 0.0
