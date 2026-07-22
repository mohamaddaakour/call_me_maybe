"""Validated data models used by the application."""

from typing import TypeAlias

from pydantic import BaseModel, ConfigDict, Field, field_validator

JsonScalar: TypeAlias = str | int | float | bool | None

SUPPORTED_SCALAR_TYPES = frozenset({"boolean", "integer", "number", "string"})


class StrictModel(BaseModel):
    """Base model that rejects fields not declared by the input schema."""

    model_config = ConfigDict(extra="forbid")


class ValueDefinition(StrictModel):
    """Describe the JSON type of a parameter or return value."""

    type: str

    @field_validator("type")
    @classmethod
    def validate_supported_type(cls, value: str) -> str:
        """Reject types outside the mandatory scalar scope.

        Args:
            value: JSON type name supplied by the definitions file.

        Returns:
            The validated type name.

        Raises:
            ValueError: If the type is empty or unsupported.
        """
        normalized = value.strip()

        if normalized not in SUPPORTED_SCALAR_TYPES:
            supported = ", ".join(sorted(SUPPORTED_SCALAR_TYPES))

            raise ValueError(
                f"unsupported type {value}; supported types: {supported}"
            )
        
        return normalized


class FunctionDefinition(StrictModel):
    """Describe one function available to the language model."""

    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    parameters: dict[str, ValueDefinition]
    returns: ValueDefinition

    @field_validator("name", "description")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        """Reject whitespace-only names and descriptions.

        Args:
            value: Text supplied by the function definition.

        Returns:
            The unchanged non-blank text.

        Raises:
            ValueError: If the text contains only whitespace.
        """
        if not value.strip():
            raise ValueError("must not be blank")
        return value

    @field_validator("parameters")
    @classmethod
    def reject_blank_parameter_names(
        cls, value: dict[str, ValueDefinition]
    ) -> dict[str, ValueDefinition]:
        """Ensure every parameter has a usable name.

        Args:
            value: Mapping of parameter names to their definitions.

        Returns:
            The validated mapping.

        Raises:
            ValueError: If a parameter name is empty or whitespace-only.
        """
        if any(not name.strip() for name in value):
            raise ValueError("parameter names must not be blank")
        return value


class PromptRecord(StrictModel):
    """Contain one natural-language request from the input workload."""

    prompt: str


class FunctionCallResult(StrictModel):
    """Represent the exact output object required by the subject."""

    prompt: str
    name: str
    parameters: dict[str, JsonScalar]
