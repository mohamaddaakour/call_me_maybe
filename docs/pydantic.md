```python
from pydantic import BaseModel, Field

# we inherited from BaseModel because I want this class as a validation class
class ParameterSchema(BaseModel):
    # ... to say this field is required
    type: JsonSchemaType = Field(..., description="JSON schema type of the parameter")

    # by default adding a field without declaring it in the validation class
    # pydantic will ignore it, but by using this we say accept and keep any additional fields
    # that are not declared in the model
    # the variable must be named model_config
    model_config = {"extra" : "allow"}
```

```python
# this will validate and return a PromptEntry object
# the model_validate receive only an object
# the return will be a regular python object
# and it is an instance of this validation class
PromptEntry.model_validate(entry)

# per example here raw is a list so I can't give the model_validate
# the raw immediately, until I put it in a dictionary
validated = FunctionDefinitions.model_validate({
    "functions": raw
})
```

```python
# reject any input fields that are not explicitly defined in the Pydantic model
model_config = {"extra": "forbid"}
```

```python
# allow Pydantic models to contain fields whose types are not known or supported by Pydantic
model_config = {"arbitrary_types_allowed": True}
```

```python
# here we puted a constraint minimum length for the name field
name: str = Field(..., min_length=1, description="Function identifier")
```

```python
parameters: Dict[str, ParameterSchema] = Field(
    # this mean if the user does not provide parameters,
    # create an empty dictionary automatically
    default_factory=dict,

    description="Mapping parameter names to their schemas"
)
```

```python
returns: Optional[ParameterSchema] = Field(
    # if we don't have returns field the default value will
    # be None
    default=None,

    description="Return type schema (this is optional)"
)
```

```python
from pydantic import model_validator

# this decorator means, after Pydantic finishes validating and creating
# all the fields of the model, run this custom validation function
@model_validator(mode="after")
    def name_must_not_be_blank(self) -> "FunctionDefinition":
        """Ensure function name is not just a whitespace only."""
        if not self.name.strip():
            raise ValueError("Function name must not be blank")

        # we return self to say: validation succeeded, keep this object
        return self
```

```python
# convert a Pydantic model object back into a normal Python dictionary

result.model_dump()
```
