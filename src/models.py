from typing import Literal, Dict, Optional, Any, List
from pydantic import BaseModel, Field, model_validator

JsonSchemaType = Literal["number", "string", "boolean", "integer", "array", "object"]


class ParameterSchema(BaseModel):
    """Schema for a single function parameter."""
    type: JsonSchemaType = Field(..., description="JSON schema type of the parameter")

    model_config = {"extra" : "allow"}


class FunctionDefinition(BaseModel):
    name: str = Field(..., min_length=1, description="Function identifier")
    description: str = Field(..., description="Human-readable description of the function")
    parameters: Dict[str, ParameterSchema] = Field(
        default_factory=dict,
        description="Mapping parameter names to their schemas"
    )
    returns: Optional[ParameterSchema] = Field(
        default=None,
        description="Return type schema (this is optional)"
    )

    @model_validator(mode="after")
    def name_must_not_be_blank(self) -> "FunctionDefinition":
        """Ensure function name is not just a whitespace only."""
        if not self.name.strip():
            raise ValueError("Function name must not be blank")
        return self


class PromptEntry(BaseModel):
    """A single natural language prompt entry from function_calling_tests.json."""

    prompt: str = Field(..., description="Natural language request to process")

    @model_validator(mode="after")
    def prompt_must_not_be_blank(self) -> "PromptEntry":
        """Ensure prompt is not empty or whitespace-only."""
        if not self.prompt.strip():
            raise ValueError("Prompt must not be blank")
        return self


class FunctionCallResult(BaseModel):
    """A single function call result written to the output JSON."""

    prompt: str = Field(..., description="The original natural language request")
    name: str = Field(..., description="Name of the function to call")
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Resolved argument values keyed by parameter name",
    )

    model_config = {"extra": "forbid"}


class FunctionDefinitions(BaseModel):
    """Validated collection of function definitions."""

    functions: List[FunctionDefinition] = Field(
        ..., description="List of available function definitions"
    )

    @model_validator(mode="after")
    def must_have_at_least_one_function(self) -> "FunctionDefinitions":
        """Ensure there is at least one function defined."""
        if len(self.functions) == 0:
            raise ValueError("functions_definition.json must contain at least one function")
        return self

    def get_by_name(self, name: str) -> Optional[FunctionDefinition]:
        """Return the FunctionDefinition with the given name, or None."""
        for fn in self.functions:
            if fn.name == name:
                return fn
        return None


class PromptList(BaseModel):
    """Validated list of prompt entries."""

    prompts: List[PromptEntry] = Field(
        ..., description="List of natural language prompts to process"
    )

class GenerationState(BaseModel):
    """Tracks the state of constrained JSON generation for one function call."""

    function_name: str = Field(..., description="The selected function name")
    current_json: str = Field(default="", description="JSON string built so far")
    generated_token_ids: List[int] = Field(
        default_factory=list,
        description="Token IDs generated so far (appended to prompt IDs)",
    )
    is_complete: bool = Field(default=False, description="Whether generation is done")

    model_config = {"arbitrary_types_allowed": True}


TokenId = int
Logit = float
TokenStr = str
VocabMapping = Dict[TokenId, TokenStr]
