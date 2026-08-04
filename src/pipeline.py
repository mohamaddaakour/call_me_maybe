"""Function-call generation pipeline."""

from __future__ import annotations

from typing import Any

from src.decoding import DecodingError, choose_function, generate_parameters
from src.models import (
    FunctionCallResult,
    FunctionCallResults,
    FunctionDefinition,
    FunctionDefinitions,
    PromptInputs,
)


def generate_calls(
    model: Any,
    definitions: FunctionDefinitions,
    prompts: PromptInputs,
) -> FunctionCallResults:
    """Generate one fully validated call for every input prompt."""
    results: list[FunctionCallResult] = []
    for prompt_item in prompts.root:
        function = choose_function(model, prompt_item.prompt, definitions.root)
        parameters = generate_parameters(model, prompt_item.prompt, function)
        _validate_parameters(function, parameters)
        results.append(
            FunctionCallResult(
                prompt=prompt_item.prompt,
                name=function.name,
                parameters=parameters,
            )
        )
    return FunctionCallResults(results)


def _validate_parameters(
    function: FunctionDefinition, parameters: dict[str, Any]
) -> None:
    """Defensively verify generated names, presence, and Python value types."""
    if set(parameters) != set(function.parameters):
        raise DecodingError(f"parameter set does not match {function.name}")
    for name, definition in function.parameters.items():
        value = parameters[name]
        valid = (
            (definition.type == "string" and isinstance(value, str))
            or (definition.type == "boolean" and isinstance(value, bool))
            or (
                definition.type == "integer"
                and isinstance(value, int)
                and not isinstance(value, bool)
            )
            or (
                definition.type == "number"
                and isinstance(value, (int, float))
                and not isinstance(value, bool)
            )
        )
        if not valid:
            raise DecodingError(
                f"parameter {name!r} does not match type {definition.type}"
            )
