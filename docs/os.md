```python
# this will return True if the path exists in our system, and false otherwise
# the path should be string
# I can give it an absolute path, or releative path but it will start searching from
# the working directory of the file I execute this line in it
os.path.exists(path)
```

```python
# this will check if the directory exists and return the path
# of this directory
output_dir = os.path.dirname(path)
```

```python
# this will make the directory
# we use exist_ok=True because by default in case the output_dir didn't exist
# this will throw an error, now we will get nothing if this happen
os.makedirs(output_dir, exist_ok=True)
```
