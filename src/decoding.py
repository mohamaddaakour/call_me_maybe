"""Generic token-level constrained decoding using only the public LLM SDK."""

from __future__ import annotations

import json
from typing import Any, Literal, cast

from src.models import FunctionDefinition, JsonType

PrefixStatus = Literal["invalid", "prefix", "complete"]
MAX_VALUE_TOKENS = 96


class DecodingError(Exception):
    """Report that the model could not produce a constrained call."""


def _encode(model: Any, text: str) -> list[int]:
    """Encode text with the SDK and normalize the returned two-dimensional tensor."""
    raw_ids = cast(list[list[int]], model.encode(text).tolist())
    if not raw_ids or not raw_ids[0]:
        raise DecodingError("the model tokenizer returned an empty token sequence")
    return [int(token_id) for token_id in raw_ids[0]]


def _chat_prompt(system: str, user: str) -> str:
    """Build a Qwen chat prompt with thinking disabled."""
    return (
        "<|im_start|>system\n"
        f"{system}<|im_end|>\n"
        "<|im_start|>user\n"
        f"{user}<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n\n</think>\n\n"
    )


def _encode_choice(model: Any, value: str) -> list[int]:
    """Encode one non-empty constrained choice."""
    raw_ids = cast(list[list[int]], model.encode(value).tolist())

    if not raw_ids or not raw_ids[0]:
        raise DecodingError(f"cannot tokenize constrained value: {value!r}")
    return [int(token_id) for token_id in raw_ids[0]]


def choose_function(
    model: Any,
    prompt: str,
    definitions: list[FunctionDefinition],
) -> FunctionDefinition:
    """Use LLM logits to choose exactly one function from the supplied catalog.

    The only allowed output paths are tokenizations of catalog function names. No
    keywords, parameter names, regular expressions, or other routing heuristics are
    used to influence the selection.
    """
    if not definitions:
        raise DecodingError("cannot choose from an empty function catalog")
    
    if len(definitions) == 1:
        return definitions[0]

    catalog = "\n".join(
        f"- {definition.name}: {definition.description}"
        for definition in definitions
    )
    
    user = (
        "Choose the single available function that best handles the request. "
        "Return only its exact function name.\n\n"
        f"Available functions:\n{catalog}\n\n"
        f"Request: {prompt}\nFunction name:"
    )

    input_ids = _encode(
        model,
        _chat_prompt("You route requests to supplied functions.", user),
    )

    choices = {
        definition.name: _encode_choice(model, definition.name)
        for definition in definitions
    }

    selected_name = _decode_choice(model, input_ids, choices)

    return next(
        definition
        for definition in definitions
        if definition.name == selected_name
    )


def _decode_choice(
    model: Any,
    input_ids: list[int],
    choices: dict[str, list[int]],
) -> str:
    """Select a value from a token-prefix trie using only model logits."""
    active = dict(choices)
    generated: list[int] = []
    while active:
        completed = [
            value for value, ids in active.items() if len(ids) == len(generated)
        ]
        if completed:
            return completed[0]
        allowed = {
            ids[len(generated)]
            for ids in active.values()
            if len(ids) > len(generated)
        }
        logits = model.get_logits_from_input_ids(input_ids + generated)
        token_id = max(allowed, key=lambda candidate: logits[candidate])
        generated.append(token_id)
        active = {
            value: ids
            for value, ids in active.items()
            if ids[: len(generated)] == generated
        }
    raise DecodingError("the model could not select an allowed value")


def generate_parameters(
    model: Any,
    request: str,
    function: FunctionDefinition,
) -> dict[str, Any]:
    """Generate every parameter independently under its declared scalar grammar."""
    parameters: dict[str, Any] = {}

    signature = ", ".join(
        f"{name}: {definition.type}"
        for name, definition in function.parameters.items()
    )

    for name, definition in function.parameters.items():
        parameters[name] = _generate_scalar(
            model,
            request,
            function,
            signature,
            name,
            definition.type,
            parameters,
        )
    return parameters


def _generate_scalar(
    model: Any,
    request: str,
    function: FunctionDefinition,
    signature: str,
    parameter_name: str,
    parameter_type: JsonType,
    previous_parameters: dict[str, Any],
) -> Any:
    """Generate one value while allowing only its declared JSON scalar type."""
    previous_json = json.dumps(previous_parameters, ensure_ascii=False)

    user = (
        f"Selected function: {function.name}({signature})\n"
        f"Description: {function.description}\n"
        f"Request: {request}\n"
        f"Values already extracted: {previous_json}\n"
        f"Extract only parameter {parameter_name!r}. Return exactly one JSON "
        f"{parameter_type} value with no label, object, markdown, or explanation.\n"
        f"JSON value for {parameter_name}:"
    )
    fixed_prefix = '"' if parameter_type == "string" else ""
    input_ids = _encode(
        model,
        _chat_prompt("You extract one typed function argument.", user)
        + fixed_prefix,
    )
    generated: list[int] = []
    for _ in range(MAX_VALUE_TOKENS):
        logits = model.get_logits_from_input_ids(input_ids + generated)
        ranked_ids = sorted(
            range(len(logits)), key=logits.__getitem__, reverse=True
        )
        previous = fixed_prefix + (
            str(model.decode(generated)) if generated else ""
        )
        for token_id in ranked_ids:
            candidate_ids = generated + [token_id]
            candidate = fixed_prefix + str(model.decode(candidate_ids))
            if candidate == previous:
                if scalar_prefix_status(
                    previous + "\n", parameter_type
                ) == "complete":
                    return _parse_scalar(previous, parameter_type)
                continue
            status = scalar_prefix_status(candidate, parameter_type)
            if status == "invalid":
                continue
            generated.append(token_id)
            if status == "complete":
                return _parse_scalar(candidate[:-1], parameter_type)
            break
        else:
            raise DecodingError(
                f"no type-valid token for parameter {parameter_name!r}"
            )
    raise DecodingError(
        f"parameter {parameter_name!r} exceeded {MAX_VALUE_TOKENS} tokens"
    )


def _parse_scalar(text: str, value_type: JsonType) -> Any:
    """Parse a completed scalar and verify its exact non-coercive JSON type."""
    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        raise DecodingError("constrained scalar was not valid JSON") from error
    valid = (
        (value_type == "string" and isinstance(value, str))
        or (value_type == "boolean" and isinstance(value, bool))
        or (
            value_type == "integer"
            and isinstance(value, int)
            and not isinstance(value, bool)
        )
        or (
            value_type == "number"
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
        )
    )
    if not valid:
        raise DecodingError(f"generated value does not match type {value_type}")
    return value


def object_prefix_status(
    text: str, function: FunctionDefinition
) -> PrefixStatus:
    """Classify a prefix of the canonical parameter object for ``function``."""
    position = 0
    items = list(function.parameters.items())
    if not items:
        return _literal_status(text, "{}\n")

    for index, (name, definition) in enumerate(items):
        literal = (
            ("{" if index == 0 else ",")
            + json.dumps(name, ensure_ascii=False)
            + ":"
        )
        literal_result = _consume_literal(text, position, literal)
        if isinstance(literal_result, str):
            return literal_result
        position = literal_result
        scalar_result = _consume_scalar(
            text,
            position,
            definition.type,
            "," if index < len(items) - 1 else "}",
        )
        if isinstance(scalar_result, str):
            return scalar_result
        position = scalar_result

    ending_result = _consume_literal(text, position, "}\n")
    if isinstance(ending_result, str):
        return ending_result
    return "complete" if ending_result == len(text) else "invalid"


def _consume_literal(
    text: str, position: int, literal: str
) -> int | PrefixStatus:
    """Consume an exact structural literal or classify its partial prefix."""
    remaining = text[position:]
    if len(remaining) < len(literal):
        return "prefix" if literal.startswith(remaining) else "invalid"
    if not remaining.startswith(literal):
        return "invalid"
    return position + len(literal)


def _consume_scalar(
    text: str,
    position: int,
    value_type: JsonType,
    delimiter: str,
) -> int | PrefixStatus:
    """Consume one typed scalar, leaving its following delimiter untouched."""
    if position == len(text):
        return "prefix"
    if value_type == "string":
        end = _json_string_end(text, position)
        if isinstance(end, str):
            return end
        return end

    delimiter_position = text.find(delimiter, position)
    if delimiter_position == -1:
        fragment = text[position:]
        return scalar_prefix_status(fragment, value_type)
    fragment = text[position:delimiter_position]
    if scalar_prefix_status(fragment + "\n", value_type) != "complete":
        return "invalid"
    return delimiter_position


def _json_string_end(text: str, position: int) -> int | PrefixStatus:
    """Find the end of a valid JSON string or classify its incomplete prefix."""
    if text[position] != '"':
        return "invalid"
    escaped = False
    unicode_digits = 0
    for index in range(position + 1, len(text)):
        character = text[index]
        if unicode_digits:
            if character not in "0123456789abcdefABCDEF":
                return "invalid"
            unicode_digits -= 1
            continue
        if escaped:
            if character == "u":
                unicode_digits = 4
            elif character not in '"\\/bfnrt':
                return "invalid"
            escaped = False
            continue
        if character == "\\":
            escaped = True
        elif character == '"':
            return index + 1
        elif ord(character) < 0x20:
            return "invalid"
    return "prefix"


def scalar_prefix_status(text: str, value_type: JsonType) -> PrefixStatus:
    """Classify a JSON scalar prefix whose required terminator is a newline."""
    if value_type == "string":
        end = _json_string_end(text, 0) if text else "prefix"
        if isinstance(end, str):
            return end
        remaining = text[end:]
        if remaining == "\n":
            return "complete"
        return "prefix" if not remaining else "invalid"
    if value_type == "boolean":
        return _literal_choices_status(text, ("true\n", "false\n"))
    return _number_prefix_status(text, value_type == "integer")


def _literal_choices_status(
    text: str, choices: tuple[str, ...]
) -> PrefixStatus:
    """Classify text against a finite set of exact literals."""
    if text in choices:
        return "complete"
    return "prefix" if any(choice.startswith(text) for choice in choices) else "invalid"


def _literal_status(text: str, literal: str) -> PrefixStatus:
    """Classify text against one exact literal."""
    if text == literal:
        return "complete"
    return "prefix" if literal.startswith(text) else "invalid"


def _number_prefix_status(text: str, integer_only: bool) -> PrefixStatus:
    """Validate a JSON number prefix followed by exactly one newline."""
    if not text:
        return "prefix"
    state = "start"
    for index, character in enumerate(text):
        if state == "start":
            if character == "-":
                state = "minus"
            elif character == "0":
                state = "zero"
            elif character in "123456789":
                state = "integer"
            else:
                return "invalid"
        elif state == "minus":
            if character == "0":
                state = "zero"
            elif character in "123456789":
                state = "integer"
            else:
                return "invalid"
        elif state in ("zero", "integer"):
            if state == "integer" and character.isdigit():
                continue
            if character == "\n":
                state = "done"
            elif not integer_only and character == ".":
                state = "dot"
            elif not integer_only and character in "eE":
                state = "exponent"
            else:
                return "invalid"
        elif state == "dot":
            if character.isdigit():
                state = "fraction"
            else:
                return "invalid"
        elif state == "fraction":
            if character.isdigit():
                continue
            if character == "\n":
                state = "done"
            elif character in "eE":
                state = "exponent"
            else:
                return "invalid"
        elif state == "exponent":
            if character in "+-":
                state = "exponent_sign"
            elif character.isdigit():
                state = "exponent_digits"
            else:
                return "invalid"
        elif state == "exponent_sign":
            if character.isdigit():
                state = "exponent_digits"
            else:
                return "invalid"
        elif state == "exponent_digits":
            if character.isdigit():
                continue
            if character == "\n":
                state = "done"
            else:
                return "invalid"
        else:
            return "invalid"
        if state == "done" and index != len(text) - 1:
            return "invalid"
    return "complete" if state == "done" else "prefix"
