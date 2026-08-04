"""Command-line entry point for the function-calling application."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]

from src.decoding import DecodingError
from src.files import InputFileError, load_model, write_model
from src.models import FunctionDefinitions, PromptInputs
from src.pipeline import generate_calls


def parse_arguments() -> argparse.Namespace:
    """Parse supported input and output path options."""
    parser = argparse.ArgumentParser(
        description="Translate prompts into schema-constrained function calls."
    )
    parser.add_argument(
        "--functions_definition",
        type=Path,
        default=Path("data/input/functions_definition.json"),
        help="path to the JSON function catalog",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/input/function_calling_tests.json"),
        help="path to the JSON prompt list",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/output/function_calling_results.json"),
        help="path for generated JSON calls",
    )
    return parser.parse_args()


def main() -> int:
    """Validate input files and create an initialized output document."""
    arguments = parse_arguments()
    try:
        definitions = load_model(
            arguments.functions_definition, FunctionDefinitions
        )
        prompts = load_model(arguments.input, PromptInputs)
        model = Small_LLM_Model()
        results = generate_calls(model, definitions, prompts)
        write_model(arguments.output, results)
    except (InputFileError, DecodingError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    except Exception as error:
        message = f"model initialization or inference failed: {error}"
        print(f"error: {message}", file=sys.stderr)
        return 1
    print(f"Wrote {len(results.root)} calls to {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
