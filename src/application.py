"""Application orchestration and command-line interface."""

from __future__ import annotations

import sys
from pathlib import Path
from argparse import ArgumentParser

from .errors import ApplicationError, GenerationNotImplementedError
from .io_handler import (
    load_function_definitions,
    load_prompt_records,
    write_results,
)

DEFAULT_FUNCTIONS_PATH = Path("data/input/functions_definition.json")
DEFAULT_INPUT_PATH = Path("data/input/function_calling_tests.json")
DEFAULT_OUTPUT_PATH = Path("data/output/function_calling_results.json")


def build_parser() -> ArgumentParser:
    """Create the command-line argument parser.

    Returns:
        A configured parser for all Phase 1 options.
    """

    parser = ArgumentParser(
        prog="call_me_maybe",
        description=(
            "description"
        )
    )

    parser.add_argument(
        "--functions_definition",
        type=Path,
        default=DEFAULT_FUNCTIONS_PATH,
        help=f"function definitions JSON (default: {DEFAULT_FUNCTIONS_PATH})",
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"prompt records JSON (default: {DEFAULT_INPUT_PATH})",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"result JSON (default: {DEFAULT_OUTPUT_PATH})",
    )

    return parser


def run(
    functions_definition_path: Path,
    input_path: Path,
    output_path: Path,
) -> None:
    """Validate input and execute the currently supported workload.

    Args:
        functions_definition_path: Available-function JSON file.
        input_path: Natural-language prompt JSON file.
        output_path: Destination for function-call results.

    Raises:
        ApplicationError: If validation, generation, or output fails.
    """
    load_function_definitions(functions_definition_path)

    prompts = load_prompt_records(input_path)

    if prompts:
        raise GenerationNotImplementedError(
            "generation is not implemented in phase 1; "
            "only empty prompt arrays can be processed"
        )
    write_results(output_path, [])


def cli(argv: list[str] | None = None) -> int:
    """Run the CLI and translate expected failures into an exit status.

    Args:
        argv: Optional arguments excluding the program name.

    Returns:
        Zero on success and one for a controlled application error.
    """
    arguments = build_parser().parse_args(argv)

    try:
        run(arguments.functions_definition, arguments.input, arguments.output)
    except ApplicationError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"wrote 0 function calls to {arguments.output}")
    return 0


def main() -> None:
    """Execute the CLI as a module and return its status to the shell."""
    raise sys.exit(cli())
