import json
from typing import List

from src.models import FunctionDefinition


def build_function_selection_prompt(
    user_prompt: str,
    functions: List[FunctionDefinition],
) -> str:
    """Build a prompt asking the LLM to choose the correct function name."""

    lines: List[str] = []

    lines.append("You are a function calling assistant.")
    lines.append(
        "Given a user request, select the single most appropriate function "
        "to call from the list below."
    )
    lines.append("Respond with ONLY the function name, nothing else.")
    lines.append("")
    lines.append("Available functions:")

    for fn in functions:
        param_summary = ", ".join(
            f"{name}: {schema.type}"
            for name, schema in fn.parameters.items()
        )
        lines.append(f"  - {fn.name}({param_summary}): {fn.description}")

    lines.append("")
    lines.append(f'User request: "{user_prompt}"')
    lines.append("")
    lines.append("Function name:")

    return "\n".join(lines)


def build_argument_extraction_prompt(
    user_prompt: str,
    function: FunctionDefinition,
) -> str:
    """Build a prompt asking the LLM to extract argument values as JSON."""

    lines: List[str] = []

    lines.append("You are a function calling assistant.")
    lines.append(
        "Extract the argument values from the user request for the given function."
    )
    lines.append("Respond with ONLY a valid JSON object containing the arguments.")
    lines.append("Do not include any explanation, just the JSON.")
    lines.append("")
    lines.append(f"Function: {function.name}")
    lines.append(f"Description: {function.description}")
    lines.append("")

    if function.parameters:
        lines.append("Parameters:")
        for param_name, param_schema in function.parameters.items():
            lines.append(f"  - {param_name} ({param_schema.type})")
        lines.append("")

    example_structure = {
        name: f"<{schema.type}>"
        for name, schema in function.parameters.items()
    }
    lines.append(f"Expected format: {json.dumps(example_structure)}")
    lines.append("")
    lines.append(f'User request: "{user_prompt}"')
    lines.append("")
    lines.append("JSON arguments:")

    return "\n".join(lines)


def get_function_name_characters(functions: List[FunctionDefinition]) -> List[str]:
    """Return all unique characters that appear in any function name."""
    chars: set = set()
    for fn in functions:
        chars.update(fn.name)
    return sorted(chars)


def format_functions_summary(functions: List[FunctionDefinition]) -> str:
    """Format a human-readable summary of all available functions."""
    
    lines: List[str] = ["Available functions:"]
    for fn in functions:
        params = ", ".join(
            f"{n}: {s.type}" for n, s in fn.parameters.items()
        )
        ret = fn.returns.type if fn.returns else "void"
        lines.append(f"  {fn.name}({params}) -> {ret}")
        lines.append(f"    {fn.description}")
    return "\n".join(lines)