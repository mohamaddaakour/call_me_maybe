import argparse
import sys
from typing import List

from src.io_handler import (
    load_function_definitions,
    load_prompts,
    write_results
)

from src.models import FunctionCallResult

DEFAULT_FUNCTIONS_DEF = "data/input/functions_definition.json"
DEFAULT_INPUT = "data/input/function_calling_tests.json"
DEFAULT_OUTPUT = "data/output/function_calls.json"

# fuction to taking the files path in the terminal
def build_arg_parser() -> argparse.ArgumentParser:
    """Construct and return the CLI argument parser."""

	# ArgumentParser is the object responsible for reading and processing command-line arguments
    parser = argparse.ArgumentParser(
        description=(
            "call me maybe — translate natural language prompts into "
            "structured function calls using constrained LLM decoding."
        ),
    )

    parser.add_argument(
        "--functions_definition",
        default=DEFAULT_FUNCTIONS_DEF,
        help=(
            f"Path to functions_definition.json "
            f"(default: {DEFAULT_FUNCTIONS_DEF})"
        ),
    )

    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help=(
            f"Path to the input prompts JSON file "
            f"(default: {DEFAULT_INPUT})"
        ),
    )

    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=(
            f"Path for the output results JSON file "
            f"(default: {DEFAULT_OUTPUT})"
        ),
    )

    return parser


def run(
    functions_definition_path: str,
    input_path: str,
    output_path: str,
) -> int:
    """Main execution pipeline."""

    function_defs = load_function_definitions(functions_definition_path)

    if function_defs is None:
        return 1

    prompts = load_prompts(input_path)
    if prompts is None:
        return 1

    print(
        "[INFO] Pipeline not yet implemented — producing placeholder output.",
        file=sys.stderr,
    )

    results: List[FunctionCallResult] = []

	# we are appending to this list that will be writed in the output file
    for entry in prompts:
        results.append(
            FunctionCallResult(
                prompt=entry.prompt,
                name="__pending__",
                parameters={},
            )
        )

    if not write_results(results, output_path):
        return 1

    return 0


def main() -> None:
    """Parse CLI arguments and execute the pipeline."""
    parser = build_arg_parser()

    args = parser.parse_args()

    exit_code = run(
        functions_definition_path=args.functions_definition,
        input_path=args.input,
        output_path=args.output,
    )

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
