"""Model access, chat prompting, and greedy decoding with and without a grammar."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, cast

from src import grammar
from src.errors import DecodingError
from src.models import FunctionDefinition, JsonType

MAX_PREVIEW_TOKENS = 24

# A function name is short, so a long one means the model is rambling
MAX_NAME_TOKENS = 24

# A scalar value is short, so a long one means the model is rambling
MAX_VALUE_TOKENS = 96


# Converting a string into a tensor array of ids, each token is an id
# the encode function will give us a tensor array of arrays
def _encode(model: Any, text: str) -> list[int]:
    """Encode text with the SDK and normalize the returned two-dimensional tensor."""
    raw_ids = cast(list[list[int]], model.encode(text).tolist())

    if not raw_ids or not raw_ids[0]:
        raise DecodingError("the model tokenizer returned an empty token sequence")

    return [int(token_id) for token_id in raw_ids[0]]


# LLM format to disable the thinking of the LLM
def _chat_prompt(system: str, user: str, assistant: str = "") -> str:
    """Build a Qwen chat prompt with thinking disabled.

    Args:
        system: The instruction that tells the model what role it plays
        user: The built prompt to get the best LLM result
        assistant: The beginning of the answer, written for the model

    The assistant part matters: generation continues from the very end of
    this text, so anything the model must continue from has to sit after
    `<|im_start|>assistant`. Left in the user turn it would be closed off by
    `<|im_end|>`, and the model would start a fresh answer instead.
    """
    return (
        "<|im_start|>system\n"
        f"{system}<|im_end|>\n"
        "<|im_start|>user\n"
        f"{user}<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n\n</think>\n\n"
        f"{assistant}"
    )


# Build the prompt to give to the LLM
def _routing_prompt(prompt: str, definitions: list[FunctionDefinition]) -> str:
    """Build the user half of the function-selection request."""
    catalog = ""

    for definition in definitions:
        catalog += f"- {definition.name}: {definition.description}\n"

    return (
        "Choose the single available function that best handles the request. "
        "Return only its exact function name.\n\n"
        f"Available functions:\n{catalog}\n\n"
        f"Request: {prompt}"
    )


def generate_unconstrained(
    model: Any,
    prompt: str,
    definitions: list[FunctionDefinition],
) -> str:
    """Greedily decode the model's raw answer with no grammar applied."""
    if not definitions:
        raise DecodingError("cannot choose from an empty function catalog")

    # We get a list of ids of the whole prompt we have to give
    # to the LLM
    input_ids = _encode(
        model,
        _chat_prompt(
            "You route requests to supplied functions.",
            _routing_prompt(prompt, definitions),
            "Function name: ",
        ),
    )

    generated: list[int] = []

    previous = ""

    for _ in range(MAX_PREVIEW_TOKENS):
        # `get_logits_from_input_ids` will predict what is the next token
        # after giving it the toekn_ids so far`
        # by giving each token a score
        # so logits is a list of scores
        logits = model.get_logits_from_input_ids(input_ids + generated)

        # Loop over the logits and get the highest element
        # return an index with highest score
        token_id = max(range(len(logits)), key=logits.__getitem__)

        # `decode()` will convert the list of ids into a text
        text = str(model.decode(generated + [token_id]))

        if text == previous:
            break

        generated.append(token_id)

        previous = text
    return previous


# Build the prompt that asks for one single parameter value
def _value_prompt(
    prompt: str,
    definition: FunctionDefinition,
    parameter_name: str,
    known: dict[str, Any],
) -> str:
    """Build the user half of the request for one parameter.

    Args:
        prompt: The original natural-language request
        definition: The function the model already selected
        parameter_name: The parameter whose value we are asking for
        known: The values decoded so far, so the model does not repeat them
    """
    parameter_type = definition.parameters[parameter_name].type

    context = ""

    # Showing the earlier values keeps the model from putting the same
    # value in every parameter
    if known:
        context = f"Values already extracted: {json.dumps(known)}\n"

    return (
        f"Request: {prompt}\n"
        f"Function: {definition.name}: {definition.description}\n"
        f"{context}"
        f"\nGive the JSON value that the request supplies for the parameter "
        f'"{parameter_name}", of type {parameter_type}. '
        f"Take it from the request as it is written there."
    )


def _next_allowed_token(
    model: Any,
    logits: list[float],
    generated: list[int],
    prefill: str,
    text: str,
    allows: Callable[[str], bool],
    complete: bool,
) -> tuple[int, str] | None:
    """Pick the best token the grammar still allows.

    Args:
        allows: Reports whether a text can still grow into a legal answer
        complete: Whether the text decoded so far is already a legal answer

    Returns:
        The chosen token and the text it produces, or None when the answer
        is finished and the model wants to move on.
    """
    while True:
        token_id = max(range(len(logits)), key=logits.__getitem__)

        # every token has been ruled out, so nothing can continue this text
        if logits[token_id] == float("-inf"):
            return None

        candidate = prefill + str(model.decode(generated + [token_id]))

        # a token that adds no text cannot carry the answer forward
        if candidate != text and allows(candidate):
            return (token_id, candidate)

        # the model prefers a token that cannot continue the text, which is
        # how it says the answer is finished
        if complete:
            return None

        # rule this token out and look at the next best one
        logits[token_id] = float("-inf")


def _decode_constrained(
    model: Any,
    prompt_text: str,
    prefill: str,
    allows: Callable[[str], bool],
    finished: Callable[[str], bool],
    max_tokens: int,
) -> str:
    """Greedily decode the highest-scoring text that stays legal throughout.

    Args:
        prompt_text: The full chat prompt, up to where generation starts
        prefill: Structure written for the model instead of generated by it
        allows: Reports whether a text can still grow into a legal answer
        finished: Reports whether a text is already a legal answer
        max_tokens: How far to go before giving up
    """
    input_ids = _encode(model, prompt_text + prefill)

    generated: list[int] = []
    text = prefill

    for _ in range(max_tokens):
        logits = model.get_logits_from_input_ids(input_ids + generated)

        chosen = _next_allowed_token(
            model, logits, generated, prefill, text, allows, finished(text)
        )

        if chosen is None:
            break

        token_id, text = chosen

        generated.append(token_id)

    return text


def generate_value(model: Any, prompt_text: str, json_type: JsonType) -> Any:
    """Decode one value that its declared JSON type allows by construction."""
    value_text = _decode_constrained(
        model,
        prompt_text,
        # The prefill is structure rather than content, so it is handed to
        # the model as part of the prompt instead of being generated
        grammar.prefill(json_type),
        lambda text: grammar.is_prefix(text, json_type),
        lambda text: grammar.is_complete(text, json_type),
        MAX_VALUE_TOKENS,
    )

    if not grammar.is_complete(value_text, json_type):
        raise DecodingError(
            f"could not decode a {json_type} value, stopped at {value_text!r}"
        )

    # The grammar already guarantees this parses, so json.loads only turns
    # the text into a Python value
    return json.loads(value_text)


def generate_selection(
    model: Any,
    prompt: str,
    definitions: list[FunctionDefinition],
) -> FunctionDefinition:
    """Decode a function name that the catalog contains by construction."""
    if not definitions:
        raise DecodingError("cannot choose from an empty function catalog")

    by_name = {definition.name: definition for definition in definitions}

    # Allowing only tokens that keep the text a prefix of some catalog name
    # is what makes an unknown or misspelled name impossible. The model still
    # makes the choice; it just cannot spell anything that does not exist.
    name = _decode_constrained(
        model,
        _chat_prompt(
            "You route requests to supplied functions.",
            _routing_prompt(prompt, definitions),
            "Function name: ",
        ),
        "",
        lambda text: any(known.startswith(text) for known in by_name),
        lambda text: text in by_name,
        MAX_NAME_TOKENS,
    )

    # Only reachable when the catalog names share a prefix and the model
    # stopped part way down it
    if name not in by_name:
        raise DecodingError(f"could not decode a function name, stopped at {name!r}")

    return by_name[name]


def generate_parameters(
    model: Any,
    prompt: str,
    definition: FunctionDefinition,
) -> dict[str, Any]:
    """Decode one value per declared parameter, in schema order."""
    parameters: dict[str, Any] = {}

    for parameter_name, parameter in definition.parameters.items():
        # Starting the answer as a JSON member puts the model exactly where
        # the value belongs, instead of at the start of a free-form reply
        prompt_text = _chat_prompt(
            "You extract function arguments from a request. You copy the "
            "values the request gives you. You never carry the function out "
            "and you never answer the request.",
            _value_prompt(prompt, definition, parameter_name, parameters),
            f'"{parameter_name}": ',
        )

        parameters[parameter_name] = generate_value(
            model, prompt_text, parameter.type
        )

    return parameters
