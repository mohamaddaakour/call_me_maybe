"""Narrow adapter around the public LLM SDK interface."""

import math
from importlib import import_module
from pathlib import Path
from typing import Any, Callable
from .errors import SDKError, TokenInspectionError
from .models import TokenInspection, Vocabulary

def initialize_model(factory: Callable[[], Any] | None = None) -> Any:
    """Initialize the default Qwen model through the supplied SDK.

    Args:
        factory: Optional test factory implementing the public SDK operations.

    Returns:
        An initialized SDK model wrapper.

    Raises:
        SDKError: If importing or initializing the wrapper fails.
    """

    try:
        if factory is None:
            # Import the module from llm_sdk
            sdk_module = import_module("llm_sdk")

            factory = sdk_module.Small_LLM_Model

        return factory()

    except Exception as error:
        # "from error" preserves the original exception
        # so Python shows both: the original cause, your custom exception
        raise SDKError(
            f"could not initialize Qwen/Qwen3-0.6B through llm_sdk: {error}"
        ) from error


def get_vocabulary_path(model: Any) -> Path:
    """Ask an SDK model for its vocabulary path.

    Args:
        model: Initialized SDK wrapper.

    Returns:
        Vocabulary filesystem path.

    Raises:
        SDKError: If the public SDK call fails or returns an invalid path.
    """
    try:
        # get_path_to_vocab_file() is a model built in method
        # to get the vocabularies path for this model
        raw_path = model.get_path_to_vocab_file()
    except Exception as error:
        raise SDKError(
            f"could not obtain the model vocabulary: {error}"
        ) from error

    if not isinstance(raw_path, (str, Path)) or not str(raw_path):
        raise SDKError("llm_sdk returned an invalid vocabulary path")
    return Path(raw_path)


def normalize_input_ids(encoded: object) -> list[int]:
    """Convert the SDK's one-row encoded value to plain token IDs."""
    try:
        raw_rows = encoded.tolist()  # type: ignore[attr-defined]
    except (AttributeError, TypeError, ValueError) as error:
        raise TokenInspectionError(
            "llm_sdk encode returned a value without a usable tolist() result"
        ) from error

    if (
            not isinstance(raw_rows, list)
            or len(raw_rows) != 1
            or not isinstance(raw_rows[0], list)
            or not raw_rows[0]
    ):
        raise TokenInspectionError(
            "llm_sdk encode must return one non-empty row of token IDs"
        )

    input_ids = raw_rows[0]

    if any(
            isinstance(token_id, bool)
            or not isinstance(token_id, int)
            or token_id < 0
            for token_id in input_ids
    ):
        raise TokenInspectionError(
            "llm_sdk encode returned an invalid token identifier"
        )
    return input_ids


def normalize_logits(raw_logits: object) -> list[float]:
    """Validate and normalize next-token logits."""
    if not isinstance(raw_logits, list) or not raw_logits:
        raise TokenInspectionError(
            "llm_sdk returned an empty or invalid logit vector"
        )

    logits: list[float] = []

    for index, value in enumerate(raw_logits):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TokenInspectionError(
                f"logit at index {index} is not numeric"
            )

        normalized = float(value)

        if math.isnan(normalized):
            raise TokenInspectionError(f"logit at index {index} is NaN")
        logits.append(normalized)
    return logits


def inspect_next_token(
        model: Any,
        prompt: str,
        vocabulary: Vocabulary,
) -> TokenInspection:
    """Encode a prompt and greedily select one vocabulary-backed token.

    Args:
        model: Initialized SDK wrapper.
        prompt: Complete model context.
        vocabulary: Validated exact token mappings.

    Returns:
        The verified greedy next-token diagnostic.

    Raises:
        SDKError: If a public inference operation fails.
        TokenInspectionError: If SDK output is structurally incompatible.
    """
    try:
        encoded = model.encode(prompt)
    except Exception as error:
        raise SDKError(
            f"could not encode the model prompt: {error}"
        ) from error

    input_ids = normalize_input_ids(encoded)

    try:
        raw_logits = model.get_logits_from_input_ids(input_ids)
    except Exception as error:
        raise SDKError(
            f"could not obtain next-token logits: {error}"
        ) from error

    logits = normalize_logits(raw_logits)

    highest_vocabulary_id = max(vocabulary.id_to_token)

    if highest_vocabulary_id >= len(logits):
        raise TokenInspectionError(
            "logit vector is incompatible with the vocabulary: "
            f"highest token ID is {highest_vocabulary_id}, but only "
            f"{len(logits)} logits were returned"
        )

    permitted_ids = vocabulary.id_to_token.keys()
    selected_id = min(
        permitted_ids,
        key=lambda token_id: (-logits[token_id], token_id),
    )
    return TokenInspection(
        input_ids=input_ids,
        token_id=selected_id,
        token_text=vocabulary.id_to_token[selected_id],
        logit=logits[selected_id],
    )
