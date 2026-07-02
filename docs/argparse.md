- The goal is to allow your program to receive options from the terminal.

```shell
example:

python -m src --input prompts.json --output results.json
```
```py
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
```
