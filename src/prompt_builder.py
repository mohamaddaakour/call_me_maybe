"""Model prompt construction from validated application inputs."""

import json
from .errors import TokenInspectionError
from .models import FunctionDefinition


def build_model_prompt(
    request: str,
    definitions: list[FunctionDefinition],
) -> str:
    """Build complete context for a next-token inspection.

    Prompt wording helps model understanding.

    Args:
        request: Original natural-language user request.
        definitions: All functions available for model selection.

    Returns:
        Text containing the request and relevant function schema details.

    Raises:
        TokenInspectionError: If no functions are available.
    """
    if not definitions:
        raise TokenInspectionError(
            "cannot inspect a function choice without function definitions"
        )

    available_functions = [
        {
            "name": definition.name,
            "description": definition.description,
            "parameters": {
                name: {"type": parameter.type}
                for name, parameter in definition.parameters.items()
            },
        }
        for definition in definitions
    ]

    serialized_functions = json.dumps(
        available_functions,

        # By default, json.dump escapes all non-ASCII characters
        # to disable that we use this
        ensure_ascii=False,

        # Remove all the spaces the output will be like that:
        # example: {"name":"search","description":"Search the web"}
        separators=(",", ":"),
    )

    return (
        "Choose one available function for the user request. Return only a "
        "function call JSON object.\n"
        f"Available functions:\n{serialized_functions}\n"
        f"User request: {request}\n"
        "Function call JSON:\n"
    )
