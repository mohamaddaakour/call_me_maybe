"""Entry point for the function calling application"""

from __future__ import annotations

import sys

from llm_sdk import Small_LLM_Model
from src.errors import DecodingError, InputFileError
from src.files import load_model, write_model
from src.models import FunctionDefinitions, PromptInputs
from src.parser import parse_arguments
from src.pipeline import generate_calls


def main() -> int:
    """Validate the input files and generate the output with matched functions"""
    arguments = parse_arguments()

    try:
        definitions = load_model(arguments.functions_definition, FunctionDefinitions)
        prompts = load_model(arguments.input, PromptInputs)

        # We called the Qwen model
        model = Small_LLM_Model()

        results = generate_calls(model, definitions, prompts)

        write_model(arguments.output, results)

    except (InputFileError, DecodingError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    # Loading the model downloads the weights and moves them onto a device,
    # so it can fail on network, disk or memory before any decoding starts
    except (OSError, RuntimeError, ValueError) as error:
        print(
            f"error: model initialization or inference failed: {error}",
            file=sys.stderr,
        )
        return 1
    print(f"Wrote {len(results.root)} calls to {arguments.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
