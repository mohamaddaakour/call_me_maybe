```python
# this is called a docstring, we use it to give a description for
# a class or function

"""Load and validate the functions_definition.json file."""
```

```python
from typing import Literal

# create a custom type called JsonSchemaType that only accepts these exact string values
JsonSchemaType = Literal["number", "string", "boolean", "integer", "array", "object"]
```

```python
# we import these classes from models.py file using
# the absolute path
from src.models import (
    FunctionCallResult,
    FunctionDefinition
)
```

```python
# this print will look like a regular print message but internally it is an error
# message

print(
    f"[Error] Function definitions file not found: {path}",
    file=sys.stderr
)
```

```python
# we use to it to verify if this variable is a list, we can use it with
# any data type also
# if the condition is correct will return True
isinstance(raw, list)
```

```python
# to print the index and value of each element in a list
# we use enumerate keyword
for index, value in enumerate(raw):
    print(index, value)
```

```python
# to take the ascii code number of this character
print(ord("!"))

# to convert from ascii code number into character
print(chr(b))
```

```python
# this will give us the value of token_id key in id_to_decoded
# if it didn't find such key it will return ""
id_to_decoded.get(token_id, "")
```