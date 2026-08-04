"""Validated models for definitions, prompts, and generated calls."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, RootModel, field_validator

JsonType = Literal["string", "number", "integer", "boolean"]


class ParameterDefinition(BaseModel):
    """Describe one required scalar function parameter."""

    model_config = ConfigDict(extra="forbid")

    type: JsonType


class ReturnDefinition(BaseModel):
    """Describe a function return type."""

    model_config = ConfigDict(extra="allow")

    type: str


class FunctionDefinition(BaseModel):
    """Describe a callable function and its required parameters."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    parameters: dict[str, ParameterDefinition]
    returns: ReturnDefinition

    @field_validator("name", "description")
    @classmethod
    def reject_empty_text(cls, value: str) -> str:
        """Reject blank function names and descriptions."""
        if not value.strip():
            raise ValueError("must not be blank")
        return value


class FunctionDefinitions(RootModel[list[FunctionDefinition]]):
    """Represent the complete non-empty function catalog."""

    @field_validator("root")
    @classmethod
    def validate_catalog(
        cls, value: list[FunctionDefinition]
    ) -> list[FunctionDefinition]:
        """Ensure the catalog is non-empty and function names are unique."""
        if not value:
            raise ValueError("at least one function definition is required")
        names = [definition.name for definition in value]
        if len(names) != len(set(names)):
            raise ValueError("function names must be unique")
        return value


class PromptInput(BaseModel):
    """Represent one natural-language function-calling request."""

    model_config = ConfigDict(extra="forbid")

    prompt: str


class PromptInputs(RootModel[list[PromptInput]]):
    """Represent the list of requests read from the input file."""


class FunctionCallResult(BaseModel):
    """Represent one schema-compliant function call."""

    model_config = ConfigDict(extra="forbid")

    prompt: str
    name: str
    parameters: dict[str, Any]


class FunctionCallResults(RootModel[list[FunctionCallResult]]):
    """Represent all calls written to the output file."""
