"""Safe JSON input and output operations."""

import json
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, TypeAdapter, ValidationError

from .errors import InputFileError, OutputFileError
from .models import FunctionCallResult, FunctionDefinition, PromptRecord

ModelType = TypeVar("ModelType", bound=BaseModel)


def read_json(path: Path) -> Any:
    """Read and decode a UTF-8 JSON file.

    Args:
        path: File to read.

    Returns:
        The decoded JSON value.

    Raises:
        InputFileError: If the file cannot be read or is not valid JSON.
    """
    try:
        with path.open("r", encoding="utf-8") as input_file:
            return json.load(input_file)
    except FileNotFoundError as error:
        raise InputFileError(f"input file not found: {path}") from error
    except PermissionError as error:
        raise InputFileError(
            f"permission denied while reading: {path}"
        ) from error
    except IsADirectoryError as error:
        raise InputFileError(
            f"expected a file but found a directory: {path}"
        ) from error
    except UnicodeDecodeError as error:
        raise InputFileError(
            f"input file is not valid UTF-8: {path}"
        ) from error
    except json.JSONDecodeError as error:
        location = f"line {error.lineno}, column {error.colno}"
        raise InputFileError(
            f"invalid JSON in {path} at {location}: {error.msg}"
        ) from error
    except OSError as error:
        raise InputFileError(f"could not read {path}: {error}") from error


def validate_list(
    raw_data: Any,
    model_type: type[ModelType],
    path: Path,
    record_label: str,
) -> list[ModelType]:
    """Validate a decoded top-level JSON array with pydantic.

    Args:
        raw_data: Decoded JSON value.
        model_type: Pydantic model required for each item.
        path: Source path used in error messages.
        record_label: Human-readable name for an array item.

    Returns:
        A list of validated models.

    Raises:
        InputFileError: If the top level or one of its records is invalid.
    """
    if not isinstance(raw_data, list):
        raise InputFileError(f"{path} must contain a top-level JSON array")
    
    try:
        # TypeAdapter is used in pydantic to validate any python data type
        # including lists elements
        adapter: TypeAdapter[list[ModelType]] = TypeAdapter(
            list[model_type]
        )

        return adapter.validate_python(raw_data)
    except ValidationError as error:
        details = error.errors(include_url=False)
        first_error = details[0]
        location = ".".join(str(part) for part in first_error["loc"])
        message = first_error["msg"]

        # we used from error because this InputFilterError happens
        # because of ValidationError
        raise InputFileError(
            f"invalid {record_label} in {path} at {location}: {message}"
        ) from error


def load_function_definitions(path: Path) -> list[FunctionDefinition]:
    """Load function definitions and reject duplicate names.

    Args:
        path: JSON file containing function definitions.

    Returns:
        Validated function definitions in input order.

    Raises:
        InputFileError: If the file or definitions are invalid.
    """
    definitions = validate_list(
        read_json(path), FunctionDefinition, path, "function definition"
    )

    seen_names: set[str] = set()

    for definition in definitions:
        if definition.name in seen_names:
            raise InputFileError(
                f"duplicate function name {definition.name!r} in {path}"
            )
        seen_names.add(definition.name)
    return definitions


def load_prompt_records(path: Path) -> list[PromptRecord]:
    """Load and validate natural-language prompt records.

    Args:
        path: JSON file containing prompt records.

    Returns:
        Validated prompt records in input order.

    Raises:
        InputFileError: If the file or records are invalid.
    """
    return validate_list(
        read_json(path), PromptRecord, path, "prompt record"
    )


def write_results(path: Path, results: list[FunctionCallResult]) -> None:
    """Write results as strict, human-readable JSON.

    Args:
        path: Destination JSON path.
        results: Fully validated output records.

    Raises:
        OutputFileError: If the destination cannot be created or written.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = [result.model_dump(mode="json") for result in results]
        
        with path.open("w", encoding="utf-8", newline="\n") as output_file:
            json.dump(payload, output_file, ensure_ascii=False, indent=2)
            output_file.write("\n")
    except (OSError, TypeError, ValueError) as error:
        raise OutputFileError(
            f"could not write output file {path}: {error}"
        ) from error
