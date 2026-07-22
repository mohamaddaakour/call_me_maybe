```python
# Reads a JSON file and converts its contents into Python objects.
json.load(input_file)
```

```python
# This writes payload (a JSON-serializable Python object, e.g. that list of dicts)
# to a file as formatted JSON text.

json.dump(payload, output_file, ensure_ascii=False, indent=2)
```