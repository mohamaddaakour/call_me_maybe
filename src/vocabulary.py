"""Loading and validation for model vocabulary files."""

import json
from pathlib import Path
from typing import cast

from .errors import VocabularyError
from .models import Vocabulary


def validate_vocabulary(raw_data: object, path: Path) -> Vocabulary:
    """Validate a token-to-identifier JSON object.

    Args:
        raw_data: Decoded vocabulary JSON.
        path: Source path used in error messages.

    Returns:
        Exact forward and reverse token mappings.

    Raises:
        VocabularyError: If the vocabulary shape or an entry is invalid.
    """
    if not isinstance(raw_data, dict):
        raise VocabularyError(
            f"vocabulary {path} must contain a top-level JSON object"
        )

    if not raw_data:
        raise VocabularyError(f"vocabulary {path} must not be empty")

    token_to_id: dict[str, int] = {}
    id_to_token: dict[int, str] = {}

    # .items() will split the keys and values
    # we used tuple unpacking to take the key as token
    # and the value as token_id
    for token, token_id in raw_data.items():

        # isinstance(token, str) will check if the value of
        # token variable is string
        if not isinstance(token, str):
            raise VocabularyError(
                f"vocabulary {path} contains a non-string token"
            )

        if (
            # Boolean are considered integers, and we don't want
            # integers for that reason we use this check
            isinstance(token_id, bool)
            or not isinstance(token_id, int)
            or token_id < 0
        ):
            raise VocabularyError(
                f"token {token!r} in {path} must have a non-negative "
                "integer ID"
            )

        # If we have duplicate
        if token_id in id_to_token:
            other_token = id_to_token[token_id]

            raise VocabularyError(
                f"duplicate token ID {token_id} in {path} for "
                f"{other_token!r} and {token!r}"
            )

        # Fill the dictionaries one is token as key and
        # and token_id as value and the other is the reverse
        # Enable the converting from token to id and vice versa
        token_to_id[token] = token_id
        id_to_token[token_id] = token

    return Vocabulary(
        token_to_id=token_to_id,
        id_to_token=id_to_token,
    )


def load_vocabulary(path: Path) -> Vocabulary:
    """Load and validate a UTF-8 token vocabulary.

    Args:
        path: JSON vocabulary path returned by the SDK.

    Returns:
        Validated exact token mappings.

    Raises:
        VocabularyError: If the file cannot be read or is invalid.
    """
    try:
        with path.open("r", encoding="utf-8") as vocabulary_file:
            raw_data = cast(object, json.load(vocabulary_file))
    except FileNotFoundError:
        raise VocabularyError(f"vocabulary file not found: {path}")
    except PermissionError:
        raise VocabularyError(
            f"permission denied while reading vocabulary: {path}"
        )
    except IsADirectoryError:
        raise VocabularyError(
            f"expected a vocabulary file but found a directory: {path}"
        )
    except UnicodeDecodeError:
        raise VocabularyError(f"vocabulary is not valid UTF-8: {path}")
    except json.JSONDecodeError as error:
        location = f"line {error.lineno}, column {error.colno}"
        raise VocabularyError(
            f"invalid vocabulary JSON in {path} at {location}: {error.msg}"
        )
    except OSError as error:
        raise VocabularyError(f"could not read vocabulary {path}: {error}")

    return validate_vocabulary(raw_data, path)
