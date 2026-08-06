"""Function-call generation pipeline."""

from __future__ import annotations

from typing import Any

from src.decoding import generate_parameters, generate_selection
from src.errors import DecodingError
from src.models import (
    FunctionCallResult,
    FunctionCallResults,
    FunctionDefinition,
    FunctionDefinitions,
    PromptInputs,
)


def _matches_type(json_type: str, value: Any) -> bool:
    """Report whether a decoded value has the type its schema declares."""
    # bool is a subclass of int in Python, so it has to be ruled out first
    if isinstance(value, bool):
        return json_type == "boolean"

    if json_type == "string":
        return isinstance(value, str)

    if json_type == "integer":
        return isinstance(value, int)

    if json_type == "number":
        return isinstance(value, (int, float))

    return False


def _validate_parameters(
    definition: FunctionDefinition,
    parameters: dict[str, Any],
) -> None:
    """Check the decoded object against the definition one last time.

    The grammar already guarantees each value parses as its declared type.
    This second pass costs nothing and turns a decoder bug into a clear
    error instead of a wrong output file.
    """
    if set(parameters) != set(definition.parameters):
        raise DecodingError(
            f"{definition.name} expects {sorted(definition.parameters)}, "
            f"decoded {sorted(parameters)}"
        )

    for parameter_name, parameter in definition.parameters.items():
        value = parameters[parameter_name]

        if not _matches_type(parameter.type, value):
            raise DecodingError(
                f"{definition.name}.{parameter_name} must be "
                f"{parameter.type}, decoded {value!r}"
            )


def generate_calls(
    model: Any,
    definitions: FunctionDefinitions,
    prompts: PromptInputs,
) -> FunctionCallResults:
    """Generate the functions calls

    Args:
        model: The LLM model
        definitions: object model that contains a list of function definition
        prompts: object model that contains a list of prompt input

    Returns:
        an object model that contain a list of function call
    """
    results: list[FunctionCallResult] = []

    for prompt_item in prompts.root:
        # The model still makes the choice, but the grammar leaves it no way
        # to name a function the catalog does not contain
        definition = generate_selection(
            model, prompt_item.prompt, definitions.root
        )

        parameters = generate_parameters(model, prompt_item.prompt, definition)

        _validate_parameters(definition, parameters)

        results.append(
            FunctionCallResult(
                prompt=prompt_item.prompt,
                name=definition.name,
                parameters=parameters,
            )
        )
    return FunctionCallResults(results)
