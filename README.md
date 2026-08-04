*This project has been created as part of the 42 curriculum by mdaakour.*

# call me maybe

## Description

`call me maybe` translates natural-language requests into typed function calls. It
loads a function catalog and a prompt list, asks the required Qwen/Qwen3-0.6B model
to select a function and extract its arguments, and writes a JSON array containing
exactly `prompt`, `name`, and `parameters` for each request.

The project uses the supplied `llm_sdk` package and only its public `encode`,
`decode`, and `get_logits_from_input_ids` methods. It does not execute the chosen
function. It implements only the mandatory scalar argument types; bonus features
such as nested arguments, extra models, batching, and a custom tokenizer are not
included.

## Instructions

Python 3.10 or newer and [uv](https://docs.astral.sh/uv/) are required. The first
run downloads Qwen/Qwen3-0.6B, so it also requires an internet connection and enough
memory for the model.

```sh
make install
make run
```

The direct equivalent is:

```sh
uv sync
uv run python -m src
```

Custom paths are supported:

```sh
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calls.json
```

The defaults are `data/input/functions_definition.json`,
`data/input/function_calling_tests.json`, and
`data/output/function_calling_results.json`. The output directory is created at
runtime and is intentionally ignored by Git.

Other mandatory Make rules are:

```sh
make debug
make clean
make lint
```

Errors such as a missing file, malformed JSON, invalid definitions, an unwritable
destination, model-loading failure, or an impossible decoding state are printed as
clear `error:` messages. The program exits nonzero instead of exposing a traceback.

## Algorithm

The input layer first validates every file with Pydantic. Definitions must be a
non-empty catalog with unique names, supported scalar parameter types, no unknown
fields, and non-blank names and descriptions. Prompt objects accept exactly one
string field.

For each prompt, function selection is performed by the LLM. All available names
and descriptions are placed in the model context. Each name is tokenized, and the
names form a prefix trie. At each generation step the program obtains the model's
logits and considers only token IDs that continue at least one name in that trie.
The highest-logit allowed token is selected. Therefore the model makes the semantic
choice, while an unknown or malformed function name is structurally impossible.

Each required argument is generated from a focused prompt built dynamically from the
selected definition. A finite-state decoder allows only its declared scalar type:

- strings enforce JSON quotes, escapes, Unicode escapes, and control-character rules;
- numbers enforce JSON sign, integer, fraction, and exponent states;
- integers disable fraction and exponent transitions;
- booleans are restricted to `true` or `false`.

The model's vocabulary is ranked from its public logits at every step. A candidate
token is decoded and accepted only when the entire accumulated text remains a valid
prefix of the declared scalar grammar. Completed values are parsed with `json.loads`,
assembled under the exact schema keys, and checked again against the chosen
definition before the Pydantic output model is serialized atomically.
The decoder contains no prompt keywords, special parameter names, value-extraction
regular expressions, or mappings for particular functions. The model chooses the
function and all values; the decoder guarantees parseability and schema compliance.

## Design decisions

- Focused scalar generation gives the small model the earlier extracted values while
  a deterministic grammar enforces every dynamic schema type.
- A trie preserves arbitrary function names and avoids keyword-based routing.
- A string's structural opening quote is prefilled; its content remains model-generated.
- Output is written through a temporary sibling file and atomically replaced, so a
  failed run does not leave truncated JSON.
- The copied SDK is excluded from lint because it is supplied third-party code; all
  project code under `src/` is checked.

## Performance and reliability

Every successful output is valid JSON and matches a selected definition by
construction. Structural reliability does not depend on the model following prose
instructions. Semantic accuracy still depends on Qwen/Qwen3-0.6B, so concise routing
and schema prompts are used to make the small model's task narrow.

A 24-prompt generalization catalog with eight unrelated functions confirmed 24/24
LLM-driven function selections. Argument semantics are probabilistic: constrained
decoding guarantees structure and types, but it cannot guarantee that a 0.6B model
copies every requested value correctly without forbidden prompt-specific rules.

Runtime depends strongly on hardware. CUDA or Apple silicon uses the SDK's reduced
precision path and is the intended way to process the supplied suite within the
five-minute target. CPU inference is substantially slower because the public SDK
recomputes the full sequence for each token and exposes no cache interface. The
implementation caps each scalar at 96 tokens so pathological generation terminates
with a clear error.

## Challenges faced

Small models often begin with explanations, repeat field labels, or emit partial
JSON. Prompt-only fixes cannot guarantee correctness. A dynamic object grammar solved
the structural problem without assumptions about particular functions or request
phrasing. A first experiment using single-letter function labels was faster but less
accurate; direct constrained names were retained because correct routing matters more
than that shortcut.

## Testing strategy

Validation covered valid example files plus malformed JSON, missing paths, duplicate
function names, unknown fields, and unsupported parameter types. Grammar checks
covered partial and complete objects, strings, escapes, booleans, negative and
exponential numbers, leading zeroes, and rejection of fractions for integers. Live
Qwen checks use multiple unrelated function catalogs and parameter names.

Before submission, run:

```sh
make lint
uv run python -m src
python -m json.tool data/output/function_calling_results.json
```

The repository contains progressive executable branches: `setup-feature`,
`validation-feature`, `constrained-decoder-feature`, and `final-solution`.

## Resources

- [JSON specification, RFC 8259](https://www.rfc-editor.org/rfc/rfc8259)
- [Pydantic documentation](https://docs.pydantic.dev/latest/)
- [Python `json` documentation](https://docs.python.org/3/library/json.html)
- [Qwen3 model documentation](https://huggingface.co/Qwen/Qwen3-0.6B)
- [uv documentation](https://docs.astral.sh/uv/)

AI was used to help review the subject requirements, explore constrained-decoding
designs, draft documentation, and propose tests. Every generated
part was implemented, reviewed, linted, type-checked, and exercised locally; an inaccurate
single-letter routing experiment and weak extraction prompts were identified through
real model tests and replaced.

End-to-end tests run demonstration prompts through Qwen/Qwen3-0.6B and independently
validate each name, required parameter set, and type against the input catalog.
