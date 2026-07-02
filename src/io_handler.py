from src.models import (
    FunctionCallResult,
    FunctionDefinition,
    FunctionDefinitions,
    PromptEntry,
)

from pydantic import ValidationError
from typing import Optional, List
import os
import sys
import json

# to load the data in function_definitions file
def load_function_definitions(path: str) -> Optional[List[FunctionDefinition]]:
    """Load and validate the functions_definition.json file."""

    if not os.path.exists(path):
        print(
            f"[Error] Function definitions file not found: {path}",
            file=sys.stderr
        )
        return None

    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
    except json.JSONDecodeError as exc:
        print(
            f"[ERROR] functions_definition.json contains invalid JSON: {exc}",
            file=sys.stderr,
        )
        return None
    except OSError as exc:
        print(
            f"[ERROR] Could not read functions_definition.json: {exc}",
            file=sys.stderr,
        )
        return None

    if not isinstance(raw, list):
        print(
            "[ERROR] functions_definition.json must be a JSON array at the top level.",
            file=sys.stderr,
        )
        return None

    try:
        # validated is an object of FunctionDefinitions pydantic model
        validated = FunctionDefinitions.model_validate({
            "functions": raw
        })
    except ValidationError as exc:
        print(
            f"[ERROR] functions_definition.json failed schema validation:\n{exc}",
            file=sys.stderr,
        )
        return None

    print(
        f"[INFO] Loaded {len(validated.functions)} function definition(s) from '{path}'."
    )
    return validated.functions


# to load the data in the prompts file
def load_prompts(path: str) -> Optional[List[PromptEntry]]:
    """Load and validate the function_calling_tests.json file."""

    if not os.path.exists(path):
        print(
            f"[ERROR] Input prompts file not found: '{path}'",
            file=sys.stderr,
        )
        return None

    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
    except json.JSONDecodeError as exc:
        print(
            f"[ERROR] Input file contains invalid JSON: {exc}",
            file=sys.stderr,
        )
        return None
    except OSError as exc:
        print(
            f"[ERROR] Could not read input file: {exc}",
            file=sys.stderr,
        )
        return None

    if not isinstance(raw, list):
        print(
            "[ERROR] Input file must be a JSON array at the top level.",
            file=sys.stderr,
        )
        return None

    prompts: List[PromptEntry] = []
    skipped = 0

    for idx, entry in enumerate(raw):
        try:
            # this will take each object and validate it than return
            # a PromptEntry object and append it to the list
            prompts.append(PromptEntry.model_validate(entry))
        except ValidationError as exc:
            # here we are not raising an error so the function will not stop
            print(
                f"[WARN] Skipping prompt at index {idx} — validation failed: {exc}",
                file=sys.stderr,
            )
            skipped += 1

    if not prompts:
        print("[ERROR] No valid prompts found in input file.", file=sys.stderr)
        return None

    if skipped:
        print(f"[WARN] Skipped {skipped} invalid prompt(s).")
    print(f"[INFO] Loaded {len(prompts)} prompt(s) from '{path}'.")

    return prompts


def write_results(results: List[FunctionCallResult], path: str) -> bool:
    """Write the list of function call results to a JSON output file."""

    output_dir = os.path.dirname(path)
    if output_dir:
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as exc:
            print(
                f"[ERROR] Could not create output directory '{output_dir}': {exc}",
                file=sys.stderr,
            )
            return False

    serialisable = [result.model_dump() for result in results]

    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(serialisable, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
    except OSError as exc:
        print(
            f"[ERROR] Could not write output file '{path}': {exc}",
            file=sys.stderr,
        )
        return False

    print(f"[INFO] Wrote {len(results)} result(s) to '{path}'.")
    return True
