"""Validation models for definitions, prompts and generated calls"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, RootModel, field_validator

# Literal means the JsonType can be one of these only
JsonType = Literal["string", "number", "integer", "boolean"]


class ParameterDefinition(BaseModel):
    """Describe one required function parameter

    Attributes:
        type: JSON type name of the parameter, e.g. "string" or "integer".
    """

    model_config = ConfigDict(extra="ignore")

    type: JsonType


class ReturnDefinition(BaseModel):
    """Describe a function return type.

    Attributes:
        type: JSON type name the function returns; extra keys are kept as-is.
    """

    model_config = ConfigDict(extra="allow")

    type: str


class FunctionDefinition(BaseModel):
    """Describe a callable function and its required parameters.

    Attributes:
        name: Unique function name used when emitting a call.
        description: Natural-language summary of what the function does.
        parameters: Parameter name to type definition; all are required.
        returns: Declared return type, when the catalog states one.
    """

    model_config = ConfigDict(extra="ignore")

    name: str
    description: str
    parameters: dict[str, ParameterDefinition]
    # The program never calls the function, so the return type is read for
    # completeness only and a catalog that omits it is still usable
    returns: ReturnDefinition | None = None

    # Validating these 2 fields name and description
    # they should not be empty
    @field_validator("name", "description")
    @classmethod
    def reject_empty_text(cls, value: str) -> str:
        """Reject blank function names and descriptions."""
        if not value.strip():
            raise ValueError("must not be blank")
        return value


# RootModel say that the entire model is a list[FunctionDefinition]
# and we can access this list using .root
class FunctionDefinitions(RootModel[list[FunctionDefinition]]):
    """Represent the complete non-empty function catalog.

    Attributes:
        root: Function definitions; non-empty, with unique names.
    """

    @field_validator("root")
    @classmethod
    def validate_catalog(
        cls, value: list[FunctionDefinition]
    ) -> list[FunctionDefinition]:
        """Ensure the catalog is non-empty and function names are unique."""
        if not value:
            raise ValueError("at least one function definition is required")

        # array containing the names of all functions
        names = [definition.name for definition in value]

        if len(names) != len(set(names)):
            raise ValueError("function names must be unique")

        return value


class PromptInput(BaseModel):
    """Represent one natural-language function-calling request.

    Attributes:
        prompt: The user's request text to resolve into a function call.
    """

    # A test file may carry its own bookkeeping next to the prompt, such as
    # an id or an expected answer, and none of it should stop the run
    model_config = ConfigDict(extra="ignore")

    prompt: str


class PromptInputs(RootModel[list[PromptInput]]):
    """Represent the list of requests read from the input file.

    Attributes:
        root: List of prompt entries, in input-file order.
    """


class FunctionCallResult(BaseModel):
    """Represent one schema-compliant function call.

    Attributes:
        prompt: The original request text this call was generated from.
        name: Name of the selected function from the catalog.
        parameters: Argument values by name, typed as the schema declares.
    """

    model_config = ConfigDict(extra="forbid")

    prompt: str
    name: str
    parameters: dict[str, Any]


class FunctionCallResults(RootModel[list[FunctionCallResult]]):
    """Represent all calls written to the output file.

    Attributes:
        root: List of generated calls, one per input prompt.
    """
