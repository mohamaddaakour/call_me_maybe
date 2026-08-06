"""Parse the input and output files path from the command line"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_arguments() -> argparse.Namespace:
    """Pasrse input and output paths from command line"""
    parser = argparse.ArgumentParser(
        description="Translate prompts into mathced function calls"
    )

    # `type=Path` will take the string from the command line and
    # apply the Path() on it to create a Path instance, so it is
    # not just a hint
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
