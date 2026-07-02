```python
# this will write in the file named fh the serialisable content
# indent=2 will make the output clear and readable
# ensure_ascii=False ensure that the ascii will stay the same
json.dump(serialisable, fh, indent=2, ensure_ascii=False)
```

```python
# reads JSON data from a file object (fh) and parses it into a Python object
json.load(fh)
```
