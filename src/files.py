"""The operations to read and write in/from a JSON file"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from src.errors import InputFileError

# We create a generic pydantic model type
ModelT = TypeVar("ModelT", bound=BaseModel)


def load_model(path: Path, model_type: type[ModelT]) -> ModelT:
    """Read a JSON file and validate it using pydantic model.

    Args:
        path: The path of the file we want to read.
        model_type: The pydantic model used for validation.

    Returns:
        The validated object model.
        Example: FunctionsDefinition(...)
    """
    try:
        with path.open("r", encoding="utf-8") as input_file:
            # Read the JSON file and return a python data structure.
            content = json.load(input_file)

        # Validate the model and if it is validated create a pydantic model
        # object
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
    """Atomically write the result to this path.

    Args:
        path: Destination JSON path.
        model: Validated data to serialize.

    Raises:
        InputFileError: If the destination cannot be written.
    """
    # Write beside the target first and swap it in at the end, so a run that
    # dies halfway leaves the previous good output intact
    temporary_path = path.with_suffix(path.suffix + ".tmp")

    try:
        # Create the directory with the parent directories
        path.parent.mkdir(parents=True, exist_ok=True)

        with temporary_path.open("w", encoding="utf-8", newline="\n") as output:
            # model_dump() converts the pydantic model to a plain Python dict
            # json.dump() convert from python dict to a json format and write it
            # to the file
            json.dump(model.model_dump(), output, indent=2, ensure_ascii=False)
            output.write("\n")

        # replace() is atomic on both POSIX and Windows
        temporary_path.replace(path)
    except OSError as error:
        # The nested try keeps a failed cleanup from hiding the real error
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise InputFileError(f"cannot write {path}: {error}") from error
