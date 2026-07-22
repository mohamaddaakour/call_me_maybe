```python
# This will create the directory in the path with the sub directories (because we used parents=True)
# exist_ok=True will not raise error if this path exists before
path.parent.mkdir(parents=True, exist_ok=True)
```