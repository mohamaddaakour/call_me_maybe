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

The project decodes on the CPU, so `torch` is pinned to the CPU-only wheel index in
`pyproject.toml`. The default PyPI wheel additionally pulls the CUDA runtime
(`nvidia-*`, `triton`): 2.2 GB of downloads for code this project never executes.
Pinning it keeps the environment at roughly 1.1 GB installed.

```sh
make install
make run
```

The direct equivalent is:

```sh
uv sync
uv run python -m src
```

On a machine whose home directory is too small for that environment, point `.venv`
at a volume that has room before syncing. `uv` follows the link and installs on the
far side of it, and the model cache can be moved the same way with `HF_HOME`:

```sh
mkdir -p /path/with/room/call-me-maybe
ln -sfn /path/with/room/call-me-maybe .venv
uv sync
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

Errors that make the run impossible — a missing file, malformed JSON, invalid
definitions, an unwritable destination, a model that fails to load — are printed as
clear `error:` messages and exit nonzero instead of exposing a traceback. A single
prompt the decoder cannot resolve is not one of them: it is reported as a
`warning:` on stderr, filled with empty values of its declared types, and the run
continues so the output file still holds one valid object per prompt.

## Algorithm

The input layer first validates every file with Pydantic. Definitions must be a
non-empty catalog with unique names, non-blank names and descriptions, and a
supported scalar type for every parameter. An unsupported parameter type is
rejected loudly rather than decoded under the wrong grammar. Documentation keys
the program does not need — a per-parameter `description`, a `required` list, a
missing `returns` — are ignored, so a catalog richer than the example still runs.
Prompt entries need a `prompt` string and may carry their own extra keys.

For each prompt, function selection is performed by the LLM. All available names
and descriptions are placed in the model context. At each generation step the
program obtains the model's logits and, from the highest score downwards, accepts
only a token whose decoded text keeps the answer a prefix of at least one catalog
name. Therefore the model makes the semantic choice, while an unknown or
misspelled function name is structurally impossible. As soon as one name is the
only one still matching, the rest of it is spelling rather than choice, so it is
completed without further generation.

Nothing is written for the model before it answers that turn. Prefilling the
prefix every catalog name shares — `fn_` for the sample catalog — looks like two
forward passes saved for free, but it splits the name across a token boundary the
model never meets in ordinary text, and the scores that follow it are noise: with
that prefill in place the model answered `fn_get_square_root` to *Greet shrek*.
Started from a clean assistant turn the first token it wants is `fn`, and routing
is correct on every prompt of the sample suite.

Ranking uses numpy. A step under a narrow grammar rejects hundreds of the roughly
150 000 vocabulary entries before one fits, and rescanning a Python list of scores
for the next best after each rejection costs several times the forward pass it
serves. One `argsort` per step replaces that scan and made the sample run about
three times faster with byte-identical output.

Each required argument is generated from a focused prompt built dynamically from the
selected definition. The decoder allows only its declared scalar type:

- strings enforce JSON quotes, escapes, Unicode escapes, and control-character rules;
- numbers enforce the JSON sign, integer, fraction, and exponent forms, and reject
  a leading zero;
- integers additionally reject fractions and exponents;
- booleans are restricted to `true` or `false`.

The model's vocabulary is ranked from its public logits at every step. A candidate
token is decoded and accepted only when the entire accumulated text remains a valid
prefix of the declared scalar grammar. Completed values are parsed with `json.loads`,
assembled under the exact schema keys, and checked again against the chosen
definition before the Pydantic output model is serialized atomically.
The decoder contains no prompt keywords, special parameter names, value-extraction
regular expressions, or mappings for particular functions. The model chooses the
function and all values; the decoder guarantees parseability and schema compliance.

Everything the model is free to decide — the function name and every argument
value — is produced under the constraint. The surrounding object is not generated
at all: its braces, its key names, and the set of keys are fully determined by the
selected definition, so emitting them from the schema leaves the model nothing to
get wrong and removes the tokens it would otherwise spend on punctuation. The
result is the same guarantee the subject asks for — every output parses, and every
key and type matches `functions_definition.json` — reached by generating only the
parts that carry information.

## Design decisions

- Focused scalar generation gives the small model the earlier extracted values while
  a deterministic grammar enforces every dynamic schema type.
- Constraining generation to the prefixes of real catalog names preserves arbitrary
  function names and avoids keyword-based routing.
- The routing turn is left clean rather than seeded with the shared name prefix: a
  prefix that is not a token boundary costs far more in wrong routing than it saves
  in forward passes.
- The argument turn states the task — filling one argument of a call — and forbids
  computing or transforming anything, before the request is shown. A small model
  otherwise reads the request as a question and answers it.
- A string's structural opening quote is prefilled; its content remains model-generated.
- Candidate tokens are ranked once per step with numpy instead of being re-scanned
  after every rejection, because the grammar rejects most of the vocabulary.
- A value the grammar can no longer extend ends generation immediately. A closed
  string or boolean cannot grow, so no forward pass is spent asking the model to
  confirm it is done; a number stays open, because `1` may still become `1.5`.
- A `number` is emitted as a float, matching the wider type the schema asked for.
- Input models ignore unknown keys, because the subject states the input files
  change at review time and a documented catalog must not stop the run.
- A prompt the decoder cannot resolve is reported on stderr and filled with empty
  values of the declared types, so one hard prompt cannot cost the whole output file.
- Output is written through a temporary sibling file and atomically replaced, so a
  failed run does not leave truncated JSON.
- The copied SDK is excluded from lint because it is supplied third-party code; all
  project code under `src/` is checked.

## Performance and reliability

Every successful output is valid JSON and matches a selected definition by
construction. Structural reliability does not depend on the model following prose
instructions. Semantic accuracy still depends on Qwen/Qwen3-0.6B, so concise routing
and schema prompts are used to make the small model's task narrow.

On the sample suite — 11 prompts against a catalog of 5 functions, one of them with
three parameters — function selection is correct 11 times out of 11, and 14 of the
17 argument values are exactly right, so 9 of the 11 calls are correct end to end.

Argument semantics are probabilistic: constrained decoding guarantees structure and
types, but it cannot guarantee that a 0.6B model reads every value correctly, and
no amount of grammar can, because the grammar constrains form and not meaning. All
three misses are the `regex` and `replacement` arguments of
`fn_substitute_string_with_regex`. Writing a regular expression from a description
— *all vowels* becoming `[aeiou]` — is authoring rather than extraction: the value
is nowhere in the request. Rewording the prompts so the pattern appears literally
was measured and does not fix it either; asked to copy `[0-9]+` the model returns
`0-9+`. That is the honest limit of this model size, and it is reported here rather
than hidden by trimming the catalog down to the easy functions. Every prompt that
states its values, including one carrying escaped quotes and one carrying a period
and an apostrophe, is extracted correctly.

Runtime is dominated by the SDK, which recomputes the whole sequence for every token
and exposes no cache interface, so the number of forward passes and the length of
each prompt are the only things worth optimising. Measured on a CPU-only machine
(10 threads, float32, the slowest path the SDK offers), the full sample suite runs
in **2 minutes 4 seconds** end to end, including roughly 25 seconds to load the
model — comfortably inside the five-minute budget. That CPU path is the one the
pinned wheel provides everywhere; on Apple silicon the SDK picks `mps` and its
reduced-precision path, which is faster still. Getting there was three
changes: ending generation at a closed value and at a uniquely identified function
name, ranking each step once with numpy instead of rescanning the vocabulary after
every rejection, and an argument prompt short enough to keep each pass cheap. The
same suite took 4 minutes 37 seconds before them. The implementation caps a function
name at 24 tokens and each scalar at 96, so pathological generation terminates
without stalling the run.

## Challenges faced

Small models often begin with explanations, repeat field labels, or emit partial
JSON. Prompt-only fixes cannot guarantee correctness. Deciding the object structure
from the schema and constraining only the values solved the structural problem
without assumptions about particular functions or request phrasing. A first
experiment using single-letter function labels was faster but less accurate; direct
constrained names were retained because correct routing matters more than that
shortcut.

The costliest mistake was an optimisation that looked obviously free. Every name in
a catalog usually opens the same way, so the shared prefix was written into the
assistant turn to save the passes that would spell it. Constrained decoding hid the
damage — the output was always a real function name — until the names themselves
were checked: *Greet shrek* routed to `fn_get_square_root`. Printing the ranked
vocabulary at that step showed why. After `fn_` the model's best tokens are `1`,
`gh`, `**`; from a clean turn its best token is `fn` at a score far above anything
else. A prefix that is not a token boundary destroys the distribution that follows
it, and the constraint cannot recover a choice the scores no longer contain.
Removing the prefill took routing on the sample suite from 8 of 11 to 11 of 11.

The same class of failure appeared in the arguments. Asked for the argument of
*What is the sum of 2 and 3?* the model answered `5`, and asked for the argument of
*Reverse the string 'hello'* it answered `"olleh"` — fluent, well-typed, and wrong,
because it was answering the request instead of filling a call. Naming the task and
forbidding computation before the request is ever shown moved the sample suite from
2 of 8 simple prompts correct to 8 of 8.

The last difficulty was cost. The SDK exposes no key-value cache, so every generated
token recomputes the whole sequence and the count of forward passes is the runtime.
Two measurements shaped the result: a third of the passes were spent learning that a
finished value was finished, which the closed-value and unique-name shortcuts
remove, and the search for the next legal token — a plain Python scan for the
maximum, repeated after each of hundreds of rejections — was itself costing more
than the forward pass it served, which one numpy `argsort` per step removes.

## Testing strategy

Validation covered valid example files plus malformed JSON, missing paths, duplicate
function names, blank descriptions, unsupported parameter types, and catalogs
carrying extra documentation keys. Grammar checks covered partial and complete
strings, escapes and `\uXXXX` sequences, booleans, negative and exponential numbers,
leading zeroes, and rejection of fractions for integers. Recovery was checked by
forcing a decoding failure and confirming the run still emits one schema-valid
object per prompt.

Because the reviewer supplies their own files, the decisive live check is a second
catalog that shares nothing with the sample one: `sendEmail`, `setThermostat`,
`toggle_light` and `count_words` — mixed camelCase and snake_case, no common prefix,
and `boolean` and `integer` parameters the sample catalog never exercises. All four
prompts route correctly and every argument is exact, including `on: false` inferred
from *turn off* and `degrees: 21` emitted as an integer rather than a float:

```sh
uv run python -m src \
  --functions_definition path/to/other_catalog.json \
  --input path/to/other_prompts.json \
  --output data/output/other.json
```

Prompt changes were never accepted on the strength of one example. Each candidate
routing turn and each candidate argument turn was run over the whole sample suite
and compared value by value against the previous one, which is how the two findings
above were separated from noise. Adversarial values were checked live: a name with
a period and an apostrophe (`Mrs. O'Brien`), and a request whose value contains
escaped quotes (`say \"hi\"`), both of which come back correctly escaped and parse.

Before submission, run:

```sh
make lint
uv run python -m src
python -m json.tool data/output/function_calling_results.json
```

The repository contains progressive executable branches: `phase-1-skeleton`,
`phase-2-model`, `phase-3-function-choice`, `phase-4-scalar-grammar`, and
`phase-5-full-pipeline`.

## Resources

- [JSON specification, RFC 8259](https://www.rfc-editor.org/rfc/rfc8259)
- [Pydantic documentation](https://docs.pydantic.dev/latest/)
- [Python `json` documentation](https://docs.python.org/3/library/json.html)
- [Qwen3 model documentation](https://huggingface.co/Qwen/Qwen3-0.6B)
- [uv documentation](https://docs.astral.sh/uv/)

AI was used to help review the subject requirements, explore constrained-decoding
designs, draft documentation, and propose tests. Every generated
part was implemented, reviewed, linted, type-checked, and exercised locally.

End-to-end tests run demonstration prompts through Qwen/Qwen3-0.6B and independently
validate each name, required parameter set, and type against the input catalog.
