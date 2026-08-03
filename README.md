*This project has been created as part of the 42 curriculum by mdaakour.*

# Call Me Maybe

## Description

Call Me Maybe is a Python application that will translate natural-language requests
into typed function calls. The finished project will use a small language model and
token-level constrained decoding so that every successful response is valid JSON and
matches one of the supplied function schemas.

This repository currently implements **Phase 2: SDK Adapter, Vocabulary, and Token
Inspection**. In addition to validated file I/O, it initializes Qwen/Qwen3-0.6B through
the supplied SDK, loads and validates the model vocabulary, builds complete model
context, obtains next-token logits, and greedily selects the highest-scoring token
represented by the vocabulary. Full function-call generation and constrained decoding
are deliberately not implemented in this phase.

Phase 1 accepts two JSON arrays:

- A function-definition array containing each function's name, description,
  parameters, and return type.
- A prompt array whose records contain a `prompt` string.

An empty prompt array produces `[]`. For a non-empty workload, the program performs
one real token inspection and then exits cleanly with a diagnostic containing the
selected token ID, exact vocabulary text, and logit. It explicitly reports that final
generation is not supported in Phase 2 instead of inventing results.

## Instructions

### Requirements

- Python 3.10 or later
- [uv](https://docs.astral.sh/uv/)
- GNU Make for the optional Makefile shortcuts

Install the locked dependencies:

```bash
# It installs all dependencies required by your project.
# Make my project exactly match what is written in pyproject.toml and uv.lock.

# It does several things automatically:
# Creates a virtual environment if one doesn't exist.
# Installs all packages.
# Removes packages that shouldn't be there.
# Uses the versions recorded in uv.lock.
uv sync
```

Run with the default paths:

```bash
# uv run automatically:
# activates the project's virtual environment
# makes sure the correct packages are available
# runs the command

# The -m flag means: Run this as a Python module.
# means:
# Find the module named src.
# Look for src/__main__.py.
# Execute it.
uv run python -m src
```

- `uv.lock` This file records the exact versions of every installed package.

The defaults are:

- Functions: `data/input/functions_definition.json`
- Prompts: `data/input/function_calling_tests.json`
- Output: `data/output/function_calling_results.json`

Run with explicit paths:

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/empty_prompts.json \
  --output data/output/function_calls.json
```

For a successful zero exit status in Phase 2, the prompt file must contain an empty
array because final generation is not implemented yet:

```json
[]
```

Makefile shortcuts are also available:

```bash
make install
make run
make debug
make lint
make clean
```

`make lint` runs the subject's required `flake8` and `mypy` checks. `make clean`
removes Python and development-tool caches.

## Input validation

The application reports concise errors and returns a non-zero status for missing or
unreadable files, malformed JSON, non-array top levels, missing or extra fields,
invalid records, duplicate function names, blank names, and unsupported parameter or
return types. Supported Phase 1 scalar type declarations are `string`, `number`,
`integer`, and `boolean`.

Parameter names are read from the definition file and are never hardcoded. Output
parent directories are created when necessary, and results are serialized as a JSON
array without comments or trailing commas.

## Algorithm

Phase 2 performs one diagnostic generation step. Its processing pipeline is:

1. Parse the three command-line paths.
2. Read both input files as UTF-8 JSON using context managers.
3. Require a top-level array and validate every record with strict pydantic models.
4. Reject duplicate function names and unsupported declared types.
5. For a non-empty workload, initialize the supplied SDK and obtain its vocabulary
   path using only public methods.
6. Validate the token-to-ID vocabulary and build exact forward and reverse mappings.
7. Build a prompt containing the user request and every supplied function's name,
   description, parameter names, and parameter types.
8. Encode that prompt, obtain logits, validate their compatibility with the
   vocabulary, and choose the highest-logit vocabulary-backed token.
9. Report the successful diagnostic and the explicit Phase 2 limitation.

The final constrained decoder is planned to maintain a generation state, test the
complete text of every candidate vocabulary token against the current JSON/schema
state, mask invalid logits, and select the highest-logit valid token. It will terminate
only in an accepting state and independently parse and validate the completed object.
None of that constrained-decoding behavior is claimed as implemented in Phase 2.

## Design decisions

- `src/__main__.py` is the single executable package boundary.
- `application.py` separates CLI orchestration from validation and model inspection.
- `llm_adapter.py` confines model access to `encode`,
  `get_logits_from_input_ids`, and `get_path_to_vocab_file`.
- The copied `llm_sdk` is installed as a local locked dependency.
- Strict pydantic models reject unknown fields rather than silently ignoring them.
- Domain-specific exceptions are translated to readable messages at the CLI boundary,
  avoiding tracebacks for expected user errors.
- `pathlib.Path` and context managers provide portable path and resource handling.
- The writer accepts validated result models, keeping the future output contract
  explicit even though Phase 1 only writes an empty array.

## Performance and reliability

Phase 2 performs linear input and vocabulary validation plus one model forward pass
for a non-empty workload. No function-selection accuracy measurement applies yet.
Reliability is checked through deterministic validation, valid empty output,
controlled model failures, and tests using adversarial fake SDK responses.

The mandatory final-project targets of at least 90% accuracy and a runtime below five
minutes cannot be measured until constrained generation is implemented.

## Challenges faced

The main Phase 2 challenge was keeping model integration behind the SDK's public
surface while checking that tensor encoding, logits, and vocabulary IDs agree. A
narrow adapter normalizes SDK values and translates model failures into application
errors without accessing private SDK state.

Readable validation errors were another concern. The loader reports the array index and
nested field location from the first pydantic error, giving users an actionable message
without exposing an internal traceback.

## Testing strategy

Phase 2 is checked with:

- A valid empty prompt array, followed by parsing the generated output again with
  Python's `json` module.
- Malformed JSON and non-array top-level values.
- Missing and unreadable input paths.
- Missing fields, extra fields, unsupported types, and duplicate function names.
- SDK initialization failure, corrupt vocabulary data, incompatible logits, greedy
  permitted-token selection, and complete prompt construction.
- A non-empty prompt array, which must inspect a token and stop cleanly without
  generating fake calls.
- `flake8 .` and the mandatory `mypy` command through `make lint`.

## Resources

- [Python `argparse` documentation](https://docs.python.org/3/library/argparse.html)
- [Python `json` documentation](https://docs.python.org/3/library/json.html)
- [Python `pathlib` documentation](https://docs.python.org/3/library/pathlib.html)
- [Pydantic models documentation](https://docs.pydantic.dev/latest/concepts/models/)
- [mypy documentation](https://mypy.readthedocs.io/)
- [flake8 documentation](https://flake8.pycqa.org/)
- The project subject, especially the Phase 2 checklist and README requirements.

AI was used to review the Phase 2 implementation against the subject, identify SDK
packaging and adapter issues, help exercise required error cases, and update this
README. The schemas, file I/O, CLI behavior, vocabulary mapping, and token inspection
were then validated with executable commands; AI is not used at runtime as a function
selector or as a replacement for constrained decoding.
