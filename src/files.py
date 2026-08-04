"""Safe JSON input and output operations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

ModelT = TypeVar("ModelT", bound=BaseModel)


class InputFileError(Exception):
    """Report a clear, recoverable problem with an input file."""


def load_model(path: Path, model_type: type[ModelT]) -> ModelT:
    """Read a UTF-8 JSON file and validate it as ``model_type``.

    Args:
        path: JSON file to read.
        model_type: Pydantic model used for validation.

    Returns:
        The validated model.

    Raises:
        InputFileError: If reading, parsing, or validation fails.
    """
    try:
        with path.open("r", encoding="utf-8") as input_file:
            content = json.load(input_file)
        return model_type.model_validate(content)
    except FileNotFoundError as error:
        raise InputFileError(f"input file not found: {path}") from error
    except PermissionError as error:
        raise InputFileError(f"permission denied reading: {path}") from error
    except json.JSONDecodeError as error:
        message = f"invalid JSON in {path} at line {error.lineno}"
        raise InputFileError(message) from error
    except ValidationError as error:
        raise InputFileError(f"invalid data in {path}: {error}") from error
    except OSError as error:
        raise InputFileError(f"cannot read {path}: {error}") from error


def write_model(path: Path, model: BaseModel) -> None:
    """Atomically write a Pydantic model as indented UTF-8 JSON.

    Args:
        path: Destination JSON path.
        model: Validated data to serialize.

    Raises:
        InputFileError: If the destination cannot be written.
    """
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with temporary_path.open("w", encoding="utf-8", newline="\n") as output:
            json.dump(model.model_dump(), output, indent=2, ensure_ascii=False)
            output.write("\n")
        temporary_path.replace(path)
    except OSError as error:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise InputFileError(f"cannot write {path}: {error}") from error
