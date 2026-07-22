```python
# We inherit from BaseModel to mark this as a pydantic model
class StrictModel(BaseModel):
    """Base model that rejects fields not declared by the input schema."""

    # Only accept the fields I declared. Reject any extra fields.
    model_config = ConfigDict(extra="forbid")
```

```python
# @field_validator is a Pydantic decorator that tells Pydantic:
# Before accepting the value of this field, run this function to validate or modify it

@field_validator("parameters")
    @classmethod
    def reject_blank_parameter_names(
        cls, value: dict[str, ValueDefinition]
    ) -> dict[str, ValueDefinition]:
```

```python
# That line converts a list of Pydantic model objects into plain JSON-serializable dictionaries.
payload = [result.model_dump(mode="json") for result in results]
```