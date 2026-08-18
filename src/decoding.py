"""Chat prompting and greedy decoding under a grammar."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, cast

import numpy as np

from src import grammar
from src.errors import DecodingError
from src.models import FunctionDefinition, JsonType

# A function name is short, so a long one means the model is rambling
MAX_NAME_TOKENS = 24

# A scalar value is short, so a long one means the model is rambling
MAX_VALUE_TOKENS = 96


# Converting a string into a tensor array of ids, each token is an id
# the encode function will give us a tensor array of arrays
def _encode(model: Any, text: str) -> list[int]:
    """Encode text and flatten the two-dimensional tensor the SDK returns."""
    raw_ids = cast(list[list[int]], model.encode(text).tolist())

    if not raw_ids or not raw_ids[0]:
        raise DecodingError("the tokenizer returned an empty token sequence")

    return [int(token_id) for token_id in raw_ids[0]]


# LLM format to disable the thinking of the LLM
def _chat_prompt(user: str, assistant: str = "") -> str:
    """Build a Qwen chat prompt with thinking disabled.

    Args:
        user: The built prompt to get the best LLM result
        assistant: The beginning of the answer, written for the model

    The assistant part matters: generation continues from the very end of
    this text, so anything the model must continue from has to sit after
    `<|im_start|>assistant`. Left in the user turn it would be closed off
    by `<|im_end|>`, and the model would start a fresh answer instead.

    There is no system turn. The SDK exposes no cache, so the whole prompt
    is read again for every single generated token and its length is the
    runtime. A separate turn costs its own markers to say what the user
    turn can say in the same words, so everything is said once, there.
    """
    return (
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
        "Choose the function that best handles the request. "
        "Answer with its name only.\n\n"
        f"Functions:\n{catalog}\n"
        f"Request: {prompt}"
    )


# Spell the function the way a signature does, so the model can see the
# parameter it is asked for next to the ones it is not
def _signature(definition: FunctionDefinition) -> str:
    """Render a function as a typed signature line."""
    arguments = ", ".join(
        f"{name}: {parameter.type}"
        for name, parameter in definition.parameters.items()
    )

    return f"{definition.name}({arguments})"


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

    A small model reads the request as a question and answers it: asked for
    the argument of "the sum of 2 and 3" it replies 5, and asked to reverse
    "hello" it replies "olleh". Naming the task as filling an argument, and
    forbidding computation before the request is ever shown, is what stops
    it; measured on the sample suite it is the difference between two of
    eight simple prompts correct and eight of eight.
    """
    parameter_type = definition.parameters[parameter_name].type

    context = ""

    # Showing the earlier values keeps the model from putting the same
    # value in every parameter
    if known:
        context = f"Already filled: {json.dumps(known)}\n"

    return (
        "You fill in the arguments of a function call. "
        "Copy each value literally from the request. "
        "Never compute, solve, reverse or transform anything.\n\n"
        f"Function: {_signature(definition)}\n"
        f"{definition.description}\n\n"
        f"Request: {prompt}\n"
        f"{context}"
        f'Value of the argument "{parameter_name}" ({parameter_type})?'
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

    The vocabulary holds about 150k entries, and a step under a narrow
    grammar can reject hundreds of tokens before one fits. Ranking the
    scores once with numpy, instead of rescanning a Python list for the
    next best after every rejection, is what keeps that search off the
    clock.
    """
    ranked = np.argsort(-np.asarray(logits, dtype=np.float32))

    for token_id in ranked:
        candidate = prefill + str(model.decode(generated + [int(token_id)]))

        # a token that adds no text cannot carry the answer forward
        if candidate != text and allows(candidate):
            return (int(token_id), candidate)

        # the model prefers a token that cannot continue the text, which is
        # how it says the answer is finished
        if complete:
            return None

    # every token has been ruled out, so nothing can continue this text
    return None


def _decode_constrained(
    model: Any,
    prompt_text: str,
    prefill: str,
    allows: Callable[[str], bool],
    finished: Callable[[str], bool],
    closed: Callable[[str], bool],
    max_tokens: int,
) -> str:
    """Greedily decode the highest-scoring text that stays legal throughout.

    Args:
        prompt_text: The full chat prompt, up to where generation starts
        prefill: Structure written for the model instead of generated by it
        allows: Reports whether a text can still grow into a legal answer
        finished: Reports whether a text is already a legal answer
        closed: Reports whether no legal answer extends this text
        max_tokens: How far to go before giving up

    A forward pass over the whole sequence is by far the most expensive
    step here, so a text the grammar can no longer extend ends the loop
    before one is spent asking the model to confirm what is already known.
    """
    input_ids = _encode(model, prompt_text + prefill)

    generated: list[int] = []
    text = prefill

    for _ in range(max_tokens):
        if closed(text):
            break

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
        lambda text: grammar.is_closed(text, json_type),
        MAX_VALUE_TOKENS,
    )

    if not grammar.is_complete(value_text, json_type):
        raise DecodingError(
            f"could not decode a {json_type} value, stopped at {value_text!r}"
        )

    # The grammar already guarantees this parses, so json.loads only turns
    # the text into a Python value
    value = json.loads(value_text)

    # "number" covers both in JSON, but the schema asked for the wider one,
    # so a whole value is reported as 2.0 rather than 2
    if json_type == "number":
        return float(value)

    return value


def generate_selection(
    model: Any,
    prompt: str,
    definitions: list[FunctionDefinition],
) -> FunctionDefinition:
    """Decode a function name that the catalog contains by construction."""
    if not definitions:
        raise DecodingError("cannot choose from an empty function catalog")

    by_name = {definition.name: definition for definition in definitions}

    def candidates(text: str) -> list[str]:
        """List the catalog names the decoded text could still become."""
        return [known for known in by_name if known.startswith(text)]

    # Allowing only tokens that keep the text a prefix of some catalog name
    # is what makes an unknown or misspelled name impossible. The model
    # still makes the choice; it just cannot spell anything that does not
    # exist.
    name = _decode_constrained(
        model,
        _chat_prompt(_routing_prompt(prompt, definitions)),
        # Nothing is written for the model here. Prefilling a prefix the
        # names share, such as "fn_", looks like a free saving but splits
        # the name across a token boundary the model never sees in text,
        # and its scores after it are noise: it answered fn_get_square_root
        # to "Greet shrek". Started clean, the first token it wants is
        # "fn" and the routing is right.
        "",
        lambda text: bool(candidates(text)),
        lambda text: text in by_name,
        # Once one name is the only one left, the rest of it is spelling
        # rather than choice, and the model has nothing further to decide
        lambda text: len(candidates(text)) == 1,
        MAX_NAME_TOKENS,
    )

    remaining = candidates(name)

    if len(remaining) == 1:
        name = remaining[0]

    # Only reachable when the model exhausts the token budget part way
    # down a prefix that several catalog names still share
    if name not in by_name:
        raise DecodingError(
            f"could not decode a function name, stopped at {name!r}"
        )

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
            _value_prompt(prompt, definition, parameter_name, parameters),
            f'"{parameter_name}": ',
        )

        parameters[parameter_name] = generate_value(
            model, prompt_text, parameter.type
        )

    return parameters
