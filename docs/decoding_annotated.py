"""ANNOTATED READING COPY of src/decoding.py -- do not import, do not run.

The code below is byte-for-byte identical to src/decoding.py; only comments
have been added. Open the two side by side.

Read it in EXECUTION ORDER, not top to bottom:

    __main__.main()
      -> pipeline.generate_calls()          for each prompt:
           -> choose_function()             [1]  which function?
                -> _chat_prompt()           [2]
                -> _encode()                [3]
                -> _encode_choice() x N     [4]
                -> _decode_choice()         [5]  TRIE-CONSTRAINED LOOP
           -> generate_parameters()         [6]  for each parameter:
                -> _generate_scalar()       [7]  GRAMMAR-CONSTRAINED LOOP
                     -> scalar_prefix_status()      [8]
                          -> _json_string_end()     [9]
                          -> _literal_choices_status()  [10]
                          -> _number_prefix_status()    [11]
                     -> _parse_scalar()             [12]

    NEVER REACHED (superseded object-level grammar):
        object_prefix_status, _consume_literal, _consume_scalar,
        _literal_status

THE ONE-SENTENCE SUMMARY
    Every token this system emits is chosen by taking the model's raw logits
    and picking the argmax RESTRICTED to tokens that provably keep the output
    inside a legal grammar -- a trie of function names in phase 1, a typed
    JSON scalar DFA in phase 2. The output is schema-correct BY CONSTRUCTION,
    not by retry-and-validate.

WHY ANY OF THIS EXISTS
    llm_sdk exposes exactly three primitives:
        encode(text)                  -> torch.Tensor  shape [1, N]
        decode(ids)                   -> str           (skip_special_tokens=True)
        get_logits_from_input_ids(ids)-> list[float]   next-token logits
    No .generate(), no sampling, no grammar support, and NO KV CACHE.
    So the decoder is hand-built on top of those three calls.

NOTE ON EXAMPLES
    Prompts and results are real, taken from data/input/ and data/output/.
    Token IDs are ILLUSTRATIVE PLACEHOLDERS -- the shapes and text are right,
    the exact integers were not produced by running the Qwen3 tokenizer.
"""

from __future__ import annotations
# Turns every annotation into a lazily-evaluated STRING instead of a live
# object. Two payoffs here:
#   1. `int | PrefixStatus` (PEP 604 unions) works on Python < 3.10, and
#      pyproject.toml declares requires-python = ">=3.10", so this matters.
#   2. No import-cycle risk: `FunctionDefinition` in a signature is never
#      evaluated at import time.
# Runtime cost: zero.

import json
from typing import Any, Literal, cast

from src.models import FunctionDefinition, JsonType


PrefixStatus = Literal["invalid", "prefix", "complete"]
# A THREE-valued verdict, not two. The single most important design decision
# in this file.
#
# Why not just valid/invalid? Because a partially-generated value is NEITHER.
# Watch "Beirut" get built one token at a time:
#
#     '"'          -> not valid JSON yet ... but not WRONG    -> "prefix"
#     '"Bei'       -> "prefix"
#     '"Beirut'    -> "prefix"
#     '"Beirut"'   -> "prefix"   (closed, but no \n terminator yet)
#     '"Beirut"\n' -> "complete" <- NOW we stop
#     '"Beirut"x'  -> "invalid"  <- junk after the closing quote
#
# A binary check would reject every intermediate step and the decoder could
# never emit even a first token.
#
# `Literal[...]` is a TYPE-CHECKER construct, not an enum. At runtime these
# are plain `str`, which is why every comparison below is `== "complete"`
# rather than `is Status.COMPLETE`.

MAX_VALUE_TOKENS = 96
# Hard fuse on the per-value generation loop.
#
# Even a perfectly masked model can loop forever INSIDE a legal grammar: a
# string parameter emitting "aaaaaaaa..." produces tokens that are each a
# valid "prefix", so the grammar alone will NEVER stop it. Only this counter
# does. 96 is far more than any scalar argument needs (~10-20 tokens for a
# city name or a timestamp).


class DecodingError(Exception):
    """Report that the model could not produce a constrained call."""
    # One named error type for the whole module. __main__.main() catches
    # exactly this (plus InputFileError) and prints a clean one-liner instead
    # of a traceback:
    #
    #     except (InputFileError, DecodingError) as error:
    #         print(f"error: {error}", file=sys.stderr)
    #         return 1
    #
    # Anything NOT wrapped in DecodingError falls through to the generic
    # `except Exception` arm and is reported as "model initialization or
    # inference failed". So the wrapping below is what decides which of the
    # two messages the user actually sees.


# =============================================================================
# [3] _encode -- called from choose_function and _generate_scalar
# =============================================================================
def _encode(model: Any, text: str) -> list[int]:
    """Encode text with the SDK and normalize the returned two-dimensional tensor."""
    # `model: Any` because llm_sdk ships no type stubs. Annotating it properly
    # would mean either lying to mypy or writing a Protocol.

    raw_ids = cast(list[list[int]], model.encode(text).tolist())
    # THE SHAPE PROBLEM THIS FUNCTION EXISTS TO SOLVE:
    #
    #   llm_sdk.encode() returns  torch.tensor([[151644, 8948, 198, ...]])
    #                                          ^^                      ^^
    #                                          a BATCH dimension of size 1
    #
    #   .tolist()   ->  [[151644, 8948, 198, ...]]   # list OF lists
    #   raw_ids[0]  ->   [151644, 8948, 198, ...]    # what we actually want
    #
    # Meanwhile get_logits_from_input_ids() wants a FLAT list[int] and re-adds
    # the batch axis itself. So this function is purely "unwrap the batch dim".
    #
    # `cast(T, x)` is a TYPE-CHECKER-ONLY assertion. At runtime it is literally
    # `return x` -- no conversion, no validation, no cost. It is here because
    # `model` is `Any`, so `.tolist()` returns `Any` and mypy loses the thread.
    # The cast hands the type information back. It does NOT make the claim true.

    if not raw_ids or not raw_ids[0]:
        raise DecodingError("the model tokenizer returned an empty token sequence")
    # Converts a silent time-bomb into a named error AT THE BOUNDARY. Without
    # it, an empty encode surfaces much later as an IndexError deep inside
    # _decode_choice, with a traceback pointing at the wrong place entirely.
    #
    #   `not raw_ids`     catches  []
    #   `not raw_ids[0]`  catches  [[]]

    return [int(token_id) for token_id in raw_ids[0]]
    # int() is redundant at RUNTIME (.tolist() already yields Python ints) but
    # makes the declared `-> list[int]` honest to the type checker.

# CORRECTNESS NOTE ------------------------------------------------------------
# llm_sdk calls tokenizer.encode(text, add_special_tokens=False). That flag
# only suppresses AUTOMATIC BOS/EOS insertion. Literal "<|im_start|>" text in
# the string is still matched by the tokenizer's added-token trie and becomes
# its proper single special-token id (151644), NOT the 6 tokens you would get
# if it were treated as ordinary text.
#
# So the hand-written ChatML in _chat_prompt tokenizes IDENTICALLY to what
# tokenizer.apply_chat_template() would have produced. That is the whole
# reason the hand-rolled template is safe.


# =============================================================================
# [2] _chat_prompt -- called by choose_function and _generate_scalar
# =============================================================================
def _chat_prompt(system: str, user: str) -> str:
    """Build a Qwen chat prompt with thinking disabled."""
    return (
        "<|im_start|>system\n"
        f"{system}<|im_end|>\n"
        "<|im_start|>user\n"
        f"{user}<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n\n</think>\n\n"
    )

# WHY IT LOOKS LIKE THIS ------------------------------------------------------
# This is ChatML, the format Qwen3 is instruction-tuned on. Normally you would
# get it from tokenizer.apply_chat_template(msgs, add_generation_prompt=True),
# but llm_sdk never exposes the tokenizer object -- only encode/decode/logits.
# So the template is hand-written to match.
#
# TWO NON-OBVIOUS DETAILS:
#
# (1) THE PROMPT ENDS MID-TURN.
#     There is no closing <|im_end|> after "assistant". That is the point: the
#     next token the model predicts IS the first token of the reply, which is
#     precisely the token we are about to mask.
#
# (2) `<think>\n\n</think>\n\n` IS THE LOAD-BEARING TRICK.
#     Qwen3 is a HYBRID REASONING model. Left alone it opens a <think> block
#     and emits hundreds of tokens of chain-of-thought before answering:
#
#         <think>
#         The user wants to book a flight. Looking at the catalog, ...
#         </think>
#         fn_book_flight
#
#     Under constrained decoding that is a disaster. The mask would force "fn"
#     as the very first token while the model is trying to write "The",
#     producing a low-confidence, essentially random choice.
#
#     Pre-filling an ALREADY-CLOSED, EMPTY think block puts the model PAST its
#     reasoning phase before generation begins. This is byte-for-byte what
#     Qwen's official chat template emits for enable_thinking=False.
#
# EXAMPLE -- the complete string handed to _encode for prompt #1:
#
#     <|im_start|>system
#     You route requests to supplied functions.<|im_end|>
#     <|im_start|>user
#     Choose the single available function that best handles the request.
#     Return only its exact function name.
#
#     Available functions:
#     - fn_book_flight: Book airline seats from an origin city to a ...
#     - fn_set_alarm: Set an alarm at a requested time and optionally ...
#     - fn_convert_temperature: Convert a temperature value to a ...
#     - fn_lookup_product: Look up a ...
#
#     Request: Book a flight from Beirut to Istanbul for 3 passengers
#     Function name:<|im_end|>
#     <|im_start|>assistant
#     <think>
#
#     </think>
#
#     ^ generation starts HERE


# =============================================================================
# [4] _encode_choice -- called once per catalog entry by choose_function
# =============================================================================
def _encode_choice(model: Any, value: str) -> list[int]:
    """Encode one non-empty constrained choice."""
    raw_ids = cast(list[list[int]], model.encode(value).tolist())
    if not raw_ids or not raw_ids[0]:
        raise DecodingError(f"cannot tokenize constrained value: {value!r}")
        # `!r` calls repr() instead of str(), so the value appears QUOTED:
        #     cannot tokenize constrained value: 'fn_book_flight'
        # rather than
        #     cannot tokenize constrained value: fn_book_flight
        # Standard practice when interpolating data into error text -- it makes
        # whitespace and empty strings visible instead of invisible.
    return [int(token_id) for token_id in raw_ids[0]]

# CODE SMELL ------------------------------------------------------------------
# This is byte-for-byte _encode() except for the error message. Straight
# duplication -- one function with an `error_message` parameter would do.
#
# The only defence: a failure HERE means "a CATALOG ENTRY is untokenizable",
# a genuinely different diagnosis than "the PROMPT failed to tokenize", and
# the author wanted that distinction to survive into the user-facing message.
# Debatable whether it is worth ~8 duplicated lines.


# =============================================================================
# [1] choose_function -- THE ENTRY POINT into this module
#     Called from pipeline.generate_calls(), once per input prompt.
# =============================================================================
def choose_function(
    model: Any,
    prompt: str,      # e.g. "Book a flight from Beirut to Istanbul for 3 passengers"
    definitions: list[FunctionDefinition],
) -> FunctionDefinition:
    """Use LLM logits to choose exactly one function from the supplied catalog.

    The only allowed output paths are tokenizations of catalog function names. No
    keywords, parameter names, regular expressions, or other routing heuristics are
    used to influence the selection.
    """
    # ^ The docstring is emphatic for a reason: the whole point of the exercise
    #   is that THE MODEL routes, not the code. No `if "flight" in prompt`.

    # ---- Guard 1: defensive, effectively dead -------------------------------
    if not definitions:
        raise DecodingError("cannot choose from an empty function catalog")
    # models.py::FunctionDefinitions.validate_catalog already rejects an empty
    # list at load time, so this can only fire if someone calls choose_function
    # directly. Kept because the alternative failure -- `next()` raising
    # StopIteration 25 lines below -- would be baffling to debug.

    # ---- Guard 2: a REAL optimisation ---------------------------------------
    if len(definitions) == 1:
        return definitions[0]
    # With one candidate there is nothing to decide. Skips an entire forward
    # pass. Also dodges a degenerate trie walk in _decode_choice (which would
    # still work, just pointlessly).

    # ---- Build the catalog block shown to the model -------------------------
    catalog = "\n".join(
        f"- {definition.name}: {definition.description}"
        for definition in definitions
    )
    # Generator expression straight into join() -- no throwaway list.
    #
    # EXAMPLE, from data/input/functions_definition.json:
    #
    #   - fn_book_flight: Book airline seats from an origin city to a
    #     destination city.
    #   - fn_set_alarm: Set an alarm at a requested time and optionally
    #     repeat it daily.
    #   - fn_convert_temperature: Convert a temperature value to a requested
    #     temperature scale.
    #   - fn_lookup_product: Look up a ...
    #
    # DESCRIPTIONS matter, not just names. "fn_set_alarm" vs "fn_book_flight"
    # is easy; a 0.6B model needs the prose to separate near-neighbours.

    user = (
        "Choose the single available function that best handles the request. "
        "Return only its exact function name.\n\n"
        f"Available functions:\n{catalog}\n\n"
        f"Request: {prompt}\nFunction name:"
    )
    # NOTE THE ENDING: "Function name:" with NO trailing content.
    #
    # This is a TRAILING CUE. Decoding is greedy and begins at the very next
    # token, so the prompt is deliberately cut off mid-thought to put the model
    # in exactly the state where "fn_book_flight" is the natural continuation.

    input_ids = _encode(
        model,
        _chat_prompt("You route requests to supplied functions.", user),
    )
    # --> steps into [2] _chat_prompt, then [3] _encode

    choices = {
        definition.name: _encode_choice(model, definition.name)
        for definition in definitions
    }
    # Dict comprehension -> the ALPHABET of the trie.  --> steps into [4]
    #
    # EXAMPLE (ids illustrative):
    #   {
    #     "fn_book_flight":         [8822, 27074, 62717],  # "fn" "_book" "_flight"
    #     "fn_set_alarm":           [8822,  2602, 51782],  # "fn" "_set"  "_alarm"
    #     "fn_convert_temperature": [8822, 25597, 33132],
    #     "fn_lookup_product":      [8822, 21396, 21361],
    #   }
    #
    # ALL FOUR SHARE TOKEN 8822 ("fn"). That shared stem is exactly what makes
    # this a TRIE rather than four independent options: step 1 is forced (only
    # one allowed token) and the real decision happens at step 2.

    selected_name = _decode_choice(model, input_ids, choices)
    # --> steps into [5], the trie walk. Returns e.g. "fn_book_flight".

    return next(
        definition
        for definition in definitions
        if definition.name == selected_name
    )
    # `next(<genexpr>)` is the Python idiom for "find the first match".
    # It raises StopIteration on no match -- impossible here, because
    # selected_name came out of `choices`, whose keys came from `definitions`,
    # and models.py::validate_catalog guarantees names are unique.


# =============================================================================
# [5] _decode_choice -- THE TRIE WALK (constrained decoder #1 of 2)
# =============================================================================
def _decode_choice(
    model: Any,
    input_ids: list[int],           # tokenized prompt; FIXED for the whole walk
    choices: dict[str, list[int]],  # name -> its token-id sequence
) -> str:
    """Select a value from a token-prefix trie using only model logits."""

    active = dict(choices)
    # SHALLOW copy. The loop rebinds `active` by filtering and the caller's
    # `choices` must survive intact. Shallow is safe: the inner lists are only
    # ever read, never mutated.

    generated: list[int] = []
    # Ids committed so far. Also doubles as the trie CURSOR -- its LENGTH is
    # the current depth, which is why `len(generated)` appears everywhere below.

    while active:
        # ---- (a) has any branch finished? -----------------------------------
        completed = [
            value for value, ids in active.items() if len(ids) == len(generated)
        ]
        if completed:
            return completed[0]
        # A branch whose FULL sequence has been emitted is a winner.
        # (`value` is an unfortunate name -- it holds the FUNCTION NAME.)
        #
        # !! LIMITATION worth knowing: if one name's tokenization is a strict
        #    PREFIX of another's -- say "fn_lookup" and "fn_lookup_product" --
        #    the shorter one wins the instant it completes and the longer
        #    becomes UNREACHABLE, no matter how confident the model is.
        #    Nothing here detects that. The current catalog has no such pair,
        #    so it is latent, not live.

        # ---- (b) build the mask ---------------------------------------------
        allowed = {
            ids[len(generated)]
            for ids in active.values()
            if len(ids) > len(generated)
        }
        # Set comprehension: the next legal token id from every surviving branch.
        #
        # EXAMPLE, depth 0 (generated == []):
        #     all four names start with 8822 ("fn") -> allowed == {8822}
        #     ONE option. The choice is forced; no real decision yet.
        #
        # EXAMPLE, depth 1 (generated == [8822]):
        #     allowed == {27074, 2602, 25597, 21396}
        #                _book   _set  _convert _lookup
        #     FOUR options. THIS is where the routing decision actually happens.

        # ---- (c) constrained argmax -----------------------------------------
        logits = model.get_logits_from_input_ids(input_ids + generated)
        token_id = max(allowed, key=lambda candidate: logits[candidate])
        # *** THE CORE MOVE OF THE ENTIRE FILE ***
        # Ask the model for all ~151k logits, then take the argmax over a
        # handful of allowed ids. Everything else is masked out of existence.
        #
        # EXAMPLE at depth 1 for "Book a flight from Beirut...":
        #     logits[27074] =  14.2   # "_book"     <- highest AMONG ALLOWED
        #     logits[ 2602] =   3.1   # "_set"
        #     logits[25597] =   2.7   # "_convert"
        #     logits[21396] =   5.9   # "_lookup"
        #     logits[ 3838] =  18.6   # " What"     <- HIGHER, but not allowed;
        #                             #                never even looked at.
        #     -> token_id = 27074
        #
        # WHY RAW LOGITS AND NO SOFTMAX?
        #     softmax is MONOTONIC -- it preserves ordering, so
        #         argmax(softmax(x)[mask]) == argmax(x[mask])
        #     Normalising would be arithmetic you immediately discard.
        #
        # WHY `input_ids + generated`?
        #     List concatenation -> a brand-new list every step. It is the ONLY
        #     way to advance state with this SDK: get_logits_from_input_ids
        #     takes the WHOLE prefix, runs a full forward pass, and returns just
        #     the last position's logits. THERE IS NO KV CACHE, so decoding n
        #     tokens costs O(n^2) attention work. An SDK limitation, not a code
        #     choice -- the module cannot route around it.

        # ---- (d) commit and prune -------------------------------------------
        generated.append(token_id)
        active = {
            value: ids
            for value, ids in active.items()
            if ids[: len(generated)] == generated
        }
        # `ids[:k]` is a SLICE -- the first k elements. Keep only branches whose
        # prefix still matches everything emitted.
        #
        # EXAMPLE after committing 27074 (generated == [8822, 27074]):
        #   "fn_book_flight"         [8822,27074,62717][:2] == [8822,27074] KEEP
        #   "fn_set_alarm"           [8822, 2602,51782][:2] != ...          DROP
        #   "fn_convert_temperature" ...                                    DROP
        #   "fn_lookup_product"      ...                                    DROP
        #   -> active == {"fn_book_flight": [...]}

    raise DecodingError("the model could not select an allowed value")
    # UNREACHABLE IN PRACTICE. token_id always comes from `allowed`, which is
    # built from active branches, so at least one branch always survives the
    # prune -- `active` can never empty out. Pure defensive backstop.

# CONTROL FLOW SHAPE ----------------------------------------------------------
# Iterative, NOT recursive. The trie is walked breadth-first-BY-DEPTH: all
# branches advance in lockstep and the candidate set narrows monotonically.
# The "branching" of the algorithm lives entirely in the `active` dict, never
# in the call stack. Divergence and reconvergence are just dict filtering.
#
# FULL TRACE for "Book a flight from Beirut to Istanbul for 3 passengers":
#
#   depth  active            allowed                   chosen
#   -----  ----------------  ------------------------  ------------------------
#     0    all 4             {8822}                    8822   "fn"
#     1    all 4             {27074,2602,25597,21396}  27074  "_book"
#     2    {fn_book_flight}  {62717}                   62717  "_flight"
#     3    {fn_book_flight}  -- `completed` fires --   return "fn_book_flight"
#
# 3 forward passes. TWO OF THEM HAD ZERO FREEDOM.


# =============================================================================
# [6] generate_parameters -- called from pipeline right after choose_function
# =============================================================================
def generate_parameters(
    model: Any,
    request: str,                  # the ORIGINAL user prompt, re-shown each time
    function: FunctionDefinition,  # the winner from choose_function
) -> dict[str, Any]:
    """Generate every parameter independently under its declared scalar grammar."""
    parameters: dict[str, Any] = {}

    signature = ", ".join(
        f"{name}: {definition.type}"
        for name, definition in function.parameters.items()
    )
    # EXAMPLE for fn_book_flight:
    #     "origin: string, destination: string, passengers: integer"
    #
    # Built ONCE and reused for every parameter, so each per-parameter prompt
    # shows the value in the context of its SIBLINGS. That is what lets the
    # model understand `origin` and `destination` are a PAIR and should not
    # both be "Beirut".

    for name, definition in function.parameters.items():
        parameters[name] = _generate_scalar(
            model,
            request,
            function,
            signature,
            name,
            definition.type,
            parameters,        # <-- the ACCUMULATOR, passed BY REFERENCE
        )
    return parameters

# THE KEY ARCHITECTURAL DECISION ----------------------------------------------
# EACH PARAMETER GETS ITS OWN INDEPENDENT INFERENCE PASS.
# The model is never asked to produce {"origin":"Beirut","destination":...} as
# one object. It is asked three separate times for three separate scalars.
#
# WHY:
#   + The grammar per call collapses to "one JSON scalar of type T" -- a small
#     character-level DFA instead of a full object grammar.
#   + Errors localise: a failure names the exact parameter that broke.
#   + The prompt can restate THAT parameter's type explicitly, which a 0.6B
#     model genuinely needs.
#
# COST:
#   - N prompt+generation sequences instead of 1 (and no KV cache, remember).
#   - No joint reasoning across parameters.
#
# The cost is partly bought back by threading `parameters` in as
# `previous_parameters`: each pass SEES what has already been extracted.
# That is SEQUENTIAL CONDITIONING done through the PROMPT rather than through
# the model's own context window:
#
#     pass 1 (origin):       previous == {}
#     pass 2 (destination):  previous == {"origin": "Beirut"}
#     pass 3 (passengers):   previous == {"origin": "Beirut", "destination": ...}
#
# TWO PYTHON DETAILS ----------------------------------------------------------
# (1) ALIASING: `parameters` is passed by reference AND mutated by the very
#     statement that consumes it. Safe, because Python fully evaluates the
#     right-hand side BEFORE performing the subscript assignment -- and
#     _generate_scalar only READS the dict (json.dumps on its first line).
#
# (2) ORDER: dict preserves INSERTION ORDER (guaranteed since Python 3.7), and
#     Pydantic preserves JSON key order when parsing
#     dict[str, ParameterDefinition]. So parameters are always generated in
#     catalog-declaration order -- origin, destination, passengers.
#     Fully deterministic.


# =============================================================================
# [7] _generate_scalar -- THE CHARACTER-LEVEL DECODER (constrained decoder #2)
#     The hardest function in the file.
# =============================================================================
def _generate_scalar(
    model: Any,
    request: str,
    function: FunctionDefinition,
    signature: str,
    parameter_name: str,           # e.g. "passengers"
    parameter_type: JsonType,      # e.g. "integer"
    previous_parameters: dict[str, Any],
) -> Any:
    """Generate one value while allowing only its declared JSON scalar type."""

    previous_json = json.dumps(previous_parameters, ensure_ascii=False)
    # ensure_ascii=False keeps non-ASCII LITERAL:
    #     True  -> {"origin": "Zürich"}   (12 extra chars, more tokens)
    #     False -> {"origin": "Zurich"}  / "Zürich"    <- what we want
    # Fewer tokens, and much closer to what the model saw in training.

    user = (
        f"Selected function: {function.name}({signature})\n"
        f"Description: {function.description}\n"
        f"Request: {request}\n"
        f"Values already extracted: {previous_json}\n"
        f"Extract only parameter {parameter_name!r}. Return exactly one JSON "
        f"{parameter_type} value with no label, object, markdown, or explanation.\n"
        f"JSON value for {parameter_name}:"
    )
    # EXAMPLE -- pass 3 (`passengers`) of prompt #1:
    #
    #   Selected function: fn_book_flight(origin: string, destination: string,
    #                                     passengers: integer)
    #   Description: Book airline seats from an origin city to a destination city.
    #   Request: Book a flight from Beirut to Istanbul for 3 passengers
    #   Values already extracted: {"origin": "Beirut", "destination": "destination"}
    #   Extract only parameter 'passengers'. Return exactly one JSON integer
    #   value with no label, object, markdown, or explanation.
    #   JSON value for passengers:
    #
    # "no label, object, markdown, or explanation" is belt-and-braces. The
    # GRAMMAR already makes ```json impossible. But a prompt that FIGHTS the
    # mask produces garbage-but-legal output, so steering still pays.

    # ---- THE FORCED-QUOTE TRICK ---------------------------------------------
    fixed_prefix = '"' if parameter_type == "string" else ""
    input_ids = _encode(
        model,
        _chat_prompt("You extract one typed function argument.", user)
        + fixed_prefix,            # <-- APPENDED TO THE PROMPT, not generated
    )
    # *** Genuinely clever and easy to skim past. ***
    #
    # For a STRING parameter the opening quote is part of the PROMPT. The model
    # is now physically mid-string and cannot emit any of the things a small
    # model loves to emit:
    #
    #     The origin city is "Beirut"     <- impossible, we are already inside "
    #     ```json                         <- impossible
    #     Sure! Here you go: "Beirut"     <- impossible
    #
    # For non-string types fixed_prefix is "" and the model starts free -- but
    # the DFA masks it just as hard (see [11] _number_prefix_status).
    #
    # THE PRICE: `fixed_prefix` is NOT in `generated`, so it must be manually
    # re-prepended EVERY time the text is reconstructed. That is why both
    # `previous` and `candidate` below start with `fixed_prefix + ...`.
    # Forget it once and the checker sees `Beirut"` instead of `"Beirut"` and
    # rejects everything.

    generated: list[int] = []

    # ---- THE GENERATION LOOP ------------------------------------------------
    for _ in range(MAX_VALUE_TOKENS):
        # `_` is the conventional name for a deliberately unused loop variable.

        logits = model.get_logits_from_input_ids(input_ids + generated)

        ranked_ids = sorted(
            range(len(logits)), key=logits.__getitem__, reverse=True
        )
        # An ARGSORT: sorting INDICES by their values, descending.
        #     range(len(logits))      -> 0, 1, 2, ... 151935   (every token id)
        #     key=logits.__getitem__  -> sort id `i` by logits[i]
        #     reverse=True            -> best first
        #
        # `logits.__getitem__` is the bound method used directly as a key
        # function -- a faster equivalent of `lambda i: logits[i]` (no Python
        # stack frame per comparison).
        #
        # !! WHY THE STRATEGY DIFFERS FROM _decode_choice:
        #    There, `allowed` was a tiny KNOWN set -> one max() call sufficed.
        #    Here, "allowed" means "every token whose TEXT keeps us inside the
        #    grammar" -- you cannot enumerate that without decoding each one.
        #    So instead: rank the ENTIRE vocabulary once, then walk down until
        #    something passes. In practice the rank-0 token usually passes, so
        #    the walk is 1-2 steps. But the SORT of ~151k floats happens on
        #    EVERY token. This is the module's main non-model CPU cost.

        previous = fixed_prefix + (
            str(model.decode(generated)) if generated else ""
        )
        # The text produced so far. The conditional avoids trusting
        # model.decode([]) on the first iteration.
        #
        # EXAMPLE mid-generation of "Beirut":
        #     generated == [ids for 'Bei','rut']
        #     previous  == '"' + 'Beirut' == '"Beirut'

        # ---- THE CANDIDATE WALK: the actual masking -------------------------
        for token_id in ranked_ids:
            candidate_ids = generated + [token_id]
            candidate = fixed_prefix + str(model.decode(candidate_ids))
            # !! IT RE-DECODES THE ENTIRE SEQUENCE rather than decoding the one
            #    new token and appending. NOT wasteful paranoia -- REQUIRED for
            #    byte-level BPE.
            #
            #    Byte-level BPE can split ONE character across MULTIPLE tokens.
            #    "e-acute" is bytes C3 A9 and may tokenize as two ids. Decoding
            #    them individually gives two U+FFFD replacement chars; only
            #    decoding them TOGETHER reconstructs the character. Same for
            #    emoji, CJK, and any multi-byte codepoint.
            #
            #        WRONG:  text += decode([new_token])   # breaks on "Zürich"
            #        RIGHT:  text  = decode(all_tokens)    # what this does
            #
            #    The standard correct-but-expensive incremental detokenization
            #    pattern.

            # ---- BRANCH A: the token added NO visible text -------------------
            if candidate == previous:
                if scalar_prefix_status(
                    previous + "\n", parameter_type
                ) == "complete":
                    return _parse_scalar(previous, parameter_type)
                continue
            # WHEN DOES THE TEXT NOT GROW?
            #     When the token is a SPECIAL token -- <|im_end|> (151645) or
            #     EOS. llm_sdk.decode() uses skip_special_tokens=True, which
            #     ERASES them, so decode(generated + [151645]) == decode(generated).
            #
            # So `candidate == previous` means: THE MODEL WANTS TO STOP HERE.
            #
            # The code treats that as an OFFER, not an order. It asks:
            #     "if I terminated right now, would this be a complete value?"
            # by appending the grammar's required \n and testing.
            #
            #   EXAMPLE, accepted:
            #     previous == '"Beirut"'
            #     '"Beirut"' + '\n' -> "complete"  -> RETURN "Beirut"
            #
            #   EXAMPLE, vetoed:
            #     previous == '"Beir'      (model tried to stop mid-string)
            #     '"Beir' + '\n' -> "invalid"   (raw \n inside a JSON string)
            #     -> `continue` -> try the rank-1 token instead, forcing the
            #        model to keep writing. THE STOP IS REFUSED.
            #
            # !! THIS IS THE PRIMARY EXIT PATH. A 0.6B Qwen ends its answer with
            #    <|im_end|>, not with a bare newline. Without Branch A the
            #    decoder would stall at rank-0 forever, burn all 96 iterations,
            #    and raise.

            # ---- BRANCH B: the token added text ------------------------------
            status = scalar_prefix_status(candidate, parameter_type)
            # --> steps into [8]

            if status == "invalid":
                continue
            # *** THIS `continue` IS THE MASK. ***
            # Discard the token no matter how confident the model was, and drop
            # to the next-best.
            #
            #   EXAMPLE, parameter_type == "integer", generated == []:
            #     rank 0: ' Three' -> candidate ' Three' -> invalid -> SKIP
            #     rank 1: 'There'  -> invalid                       -> SKIP
            #     rank 2: '3'      -> candidate '3' -> "prefix"     -> ACCEPT
            #
            #   The model "wanted" to write the WORD Three; the grammar made the
            #   digit the only reachable option.

            generated.append(token_id)

            if status == "complete":
                return _parse_scalar(candidate[:-1], parameter_type)
            # `candidate[:-1]` strips EXACTLY ONE trailing character -- the \n
            # terminator that made it complete -- before handing to json.loads.
            #     candidate      == '3\n'
            #     candidate[:-1] == '3'      -> json.loads -> 3

            break
            # status == "prefix": committed but not done. Break out of the
            # candidate walk and go around the outer loop for FRESH logits at
            # the new position.

        else:
            raise DecodingError(
                f"no type-valid token for parameter {parameter_name!r}"
            )
        # !! PYTHON-SPECIFIC, EASY TO MISREAD: `for ... else`.
        #    The `else` runs ONLY IF the loop finished WITHOUT hitting `break`.
        #    Read it as "for ... THEN" or, better, "NO-BREAK".
        #
        #    Reaching it means every one of the ~151k tokens in the vocabulary
        #    was rejected -- a total constraint failure (a grammar bug, or a
        #    type nothing can satisfy).
        #
        #    `return` ALSO bypasses `else`, so both success paths (the Branch A
        #    return and the status == "complete" return) correctly skip it.
        #    Only `break` and `return` skip it; `continue` does NOT.

    raise DecodingError(
        f"parameter {parameter_name!r} exceeded {MAX_VALUE_TOKENS} tokens"
    )
    # The outer range() ran out. The model got stuck in a legal-but-endless
    # prefix, e.g. a string of 96 tokens with no closing quote.

# WORKED TRACE -- `passengers: integer`, request "...for 3 passengers" ---------
#
#   iter  generated  previous  rank  candidate  status     action
#   ----  ---------  --------  ----  ---------  ---------  ---------------------
#     0   []         ''          0   ' 3'       invalid    skip (leading space)
#     0   []         ''          1   'Three'    invalid    skip
#     0   []         ''          2   '2'        prefix     COMMIT, break
#     1   ['2']      '2'         0   '2\n'      complete   _parse_scalar('2') -> 2
#
# !! Look at the REAL data/output/function_calling_results.json:
#        "passengers":  2               (the prompt says 3)
#        "destination": "destination"   (the prompt says Istanbul)
#        "value":       22.5            (the prompt says 37.5)
#
#    THE GRAMMAR GUARANTEES *TYPE* CORRECTNESS, NOT *SEMANTIC* CORRECTNESS.
#    `2` is a perfectly legal JSON integer, so the mask has no objection.
#    Getting 2 instead of 3 is a MODEL-CAPABILITY problem (Qwen3-0.6B), not a
#    decoding problem -- fixable with better prompting or a bigger model,
#    NEVER with a tighter grammar.


# =============================================================================
# [12] _parse_scalar -- THE FINAL GATE, called from both exits of [7]
# =============================================================================
def _parse_scalar(text: str, value_type: JsonType) -> Any:
    """Parse a completed scalar and verify its exact non-coercive JSON type."""
    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        raise DecodingError("constrained scalar was not valid JSON") from error
    # *** `raise X from error` sets __cause__, so the traceback shows BOTH:
    #
    #       json.decoder.JSONDecodeError: Expecting value: line 1 column 1
    #
    #       The above exception was the direct cause of the following exception:
    #
    #       src.decoding.DecodingError: constrained scalar was not valid JSON
    #
    #     Strongly preferred over a bare `raise` when translating an exception
    #     across an abstraction boundary: the caller gets the module's own error
    #     type WITHOUT losing the underlying diagnosis.
    #     (`raise X from None` would deliberately SUPPRESS the cause.)
    #
    # *** THIS IS A SECOND, INDEPENDENT VALIDATION. The grammar already proved
    #     the text well-formed; json.loads re-proves it with the real parser.
    #     Cheap insurance against a bug in the hand-written DFA -- e.g. the
    #     .isdigit() hole flagged in [11] is caught right here.

    valid = (
        (value_type == "string" and isinstance(value, str))
        or (value_type == "boolean" and isinstance(value, bool))
        or (
            value_type == "integer"
            and isinstance(value, int)
            and not isinstance(value, bool)      # <-- *** LOAD-BEARING ***
        )
        or (
            value_type == "number"
            and isinstance(value, (int, float))
            and not isinstance(value, bool)      # <-- *** LOAD-BEARING ***
        )
    )
    # *** THE `not isinstance(value, bool)` CLAUSES ARE NOT PARANOIA. ***
    #
    #     IN PYTHON, bool IS A SUBCLASS OF int:
    #
    #         >>> isinstance(True, int)
    #         True
    #         >>> True == 1
    #         True
    #         >>> True + True
    #         2
    #
    #     Without the exclusion a boolean would happily satisfy an `integer`
    #     parameter and you would silently write {"passengers": true}.
    #     One of the most-missed gotchas in Python type checking. The same
    #     guard appears again in pipeline.py::_validate_parameters.
    #
    # *** MEANWHILE `number` ACCEPTS int OR float DELIBERATELY:
    #     JSON `3` is a perfectly valid `number`, and json.loads("3") returns a
    #     Python int, not a float. Rejecting int here would break
    #     {"value": 37} for fn_convert_temperature.

    if not valid:
        raise DecodingError(f"generated value does not match type {value_type}")
    return value

# EXAMPLES --------------------------------------------------------------------
#   _parse_scalar('"Beirut"', "string")   ->  'Beirut'
#   _parse_scalar('3',        "integer")  ->  3
#   _parse_scalar('37.5',     "number")   ->  37.5
#   _parse_scalar('37',       "number")   ->  37       (int IS a valid number)
#   _parse_scalar('true',     "boolean")  ->  True
#   _parse_scalar('true',     "integer")  ->  DecodingError   <- bool blocked
#   _parse_scalar('3.5',      "integer")  ->  DecodingError
#   _parse_scalar('"3"',      "integer")  ->  DecodingError   <- NO COERCION:
#                                             the string "3" is NOT cast to 3
#
# "non-coercive" in the docstring means exactly that: nothing is converted.
# A value that is not ALREADY the right type is an error, not a cast.


# #############################################################################
# #                                                                           #
# #   EVERYTHING FROM HERE TO scalar_prefix_status IS DEAD CODE.              #
# #                                                                           #
# #   object_prefix_status, _consume_literal, _consume_scalar and            #
# #   _literal_status have NO CALLERS anywhere in the repo, and there is     #
# #   no test suite. Together they form a COMPLETE                           #
# #   ALTERNATIVE IMPLEMENTATION: a grammar for the WHOLE parameter object    #
# #   in one constrained pass, superseded by the per-parameter approach in    #
# #   [6] generate_parameters.                                                #
# #                                                                           #
# #   Skip to [8] on a first read. Come back for the design history.          #
# #                                                                           #
# #############################################################################

def object_prefix_status(
    text: str, function: FunctionDefinition
) -> PrefixStatus:
    """Classify a prefix of the canonical parameter object for ``function``."""
    # THE ROAD NOT TAKEN. This validates the ENTIRE object in ONE constrained
    # generation:
    #
    #     {"origin":"Beirut","destination":"Istanbul","passengers":3}\n
    #
    # Note the CANONICAL form: NO SPACES ANYWHERE. Deliberate -- every
    # structural byte is forced, so the model has ZERO freedom outside the
    # scalar values themselves. Even the KEY NAMES are literals it must emit.

    position = 0                   # a CURSOR threaded through the whole scan
    items = list(function.parameters.items())
    if not items:
        return _literal_status(text, "{}\n")
        # Zero-parameter function -> the only legal output is exactly "{}\n".

    for index, (name, definition) in enumerate(items):
        # NESTED TUPLE UNPACKING in the for target: enumerate yields
        # (index, (name, definition)) and both levels destructure at once.

        literal = (
            ("{" if index == 0 else ",")
            + json.dumps(name, ensure_ascii=False)
            + ":"
        )
        # EXAMPLE for fn_book_flight:
        #     index 0 -> '{"origin":'
        #     index 1 -> ',"destination":'
        #     index 2 -> ',"passengers":'
        # json.dumps(name) rather than f'"{name}"' so a key containing a quote
        # or backslash is escaped correctly.

        literal_result = _consume_literal(text, position, literal)
        if isinstance(literal_result, str):
            return literal_result      # "prefix" or "invalid" -> bubble out
        position = literal_result      # an int -> advance the cursor

        scalar_result = _consume_scalar(
            text,
            position,
            definition.type,
            "," if index < len(items) - 1 else "}",  # what must FOLLOW the value
        )
        if isinstance(scalar_result, str):
            return scalar_result
        position = scalar_result

    ending_result = _consume_literal(text, position, "}\n")
    if isinstance(ending_result, str):
        return ending_result
    return "complete" if ending_result == len(text) else "invalid"
    # The cursor must land EXACTLY on the end of the text -- trailing junk is
    # invalid.


def _consume_literal(
    text: str, position: int, literal: str
) -> int | PrefixStatus:
    """Consume an exact structural literal or classify its partial prefix."""
    remaining = text[position:]
    if len(remaining) < len(literal):
        return "prefix" if literal.startswith(remaining) else "invalid"
        # *** THE PARTIAL-LITERAL CASE. The text ends MID-literal:
        #       literal='{"origin":'  remaining='{"or'  -> "prefix"  (on track)
        #       literal='{"origin":'  remaining='{"de'  -> "invalid" (wrong key)
        #
        #     Note the arguments are SWAPPED versus the full-match case below:
        #     here the LITERAL starts with the REMAINING, not the other way
        #     round. Getting that backwards is a classic prefix-matching bug.
    if not remaining.startswith(literal):
        return "invalid"
    return position + len(literal)


def _consume_scalar(
    text: str,
    position: int,
    value_type: JsonType,
    delimiter: str,
) -> int | PrefixStatus:
    """Consume one typed scalar, leaving its following delimiter untouched."""
    if position == len(text):
        return "prefix"            # nothing generated yet at this slot
    if value_type == "string":
        end = _json_string_end(text, position)
        if isinstance(end, str):
            return end
        return end
        # !!!! VISIBLE BUG. Both branches return the SAME thing -- the
        #      isinstance check does literally nothing. Presumably a refactor
        #      left half-finished.
        #
        #      Worse: the string branch never verifies the DELIMITER follows,
        #      unlike the numeric branch below. So '{"origin":"Beirut"X' would
        #      be accepted here and only caught later. Dormant, since nothing
        #      calls this -- but it is a real defect, not a style nit.

    delimiter_position = text.find(delimiter, position)
    if delimiter_position == -1:
        fragment = text[position:]
        return scalar_prefix_status(fragment, value_type)
        # No delimiter yet -> still mid-value -> delegate to the scalar checker.
        # EXAMPLE: text='...":3' looking for ',' -> fragment='3' -> "prefix"
    fragment = text[position:delimiter_position]
    if scalar_prefix_status(fragment + "\n", value_type) != "complete":
        return "invalid"
    # *** THE `+ "\n"` ADAPTER. scalar_prefix_status REQUIRES a newline
    #     terminator, but INSIDE an object the terminator is ',' or '}'.
    #     So the delimiter is stripped and a synthetic \n spliced on to reuse
    #     the exact same DFA. Same trick as Branch A in [7].
    #
    #     EXAMPLE: text='...":3,"passengers"...', delimiter=','
    #              fragment='3' -> '3\n' -> "complete" -> return index of ','
    return delimiter_position
    # LEAVES the delimiter in place for the next _consume_literal to eat.


# #############################################################################
# #   END OF DEAD CODE -- live path resumes below                             #
# #############################################################################


# =============================================================================
# [9] _json_string_end -- called from [8], and from the dead _consume_scalar
# =============================================================================
def _json_string_end(text: str, position: int) -> int | PrefixStatus:
    """Find the end of a valid JSON string or classify its incomplete prefix."""
    # WHY NOT JUST USE json.loads()?
    #     Because json.loads is ALL-OR-NOTHING. It cannot say "valid so far".
    #     This scanner exists specifically to produce the three-valued answer.

    if text[position] != '"':
        return "invalid"

    escaped = False       # previous char was '\' -> this one is part of an escape
    unicode_digits = 0    # COUNTDOWN: set to 4 by '\u', decremented per hex digit
    # *** Two scalars instead of a state enum. `unicode_digits` encodes four
    #     sequential states (\uX, \uXX, \uXXX, \uXXXX) in ONE integer.

    for index in range(position + 1, len(text)):
        character = text[index]

        # ORDER MATTERS -- innermost state first.
        if unicode_digits:
            if character not in "0123456789abcdefABCDEF":
                return "invalid"
            unicode_digits -= 1
            continue
        if escaped:
            if character == "u":
                unicode_digits = 4
            elif character not in '"\\/bfnrt':
                return "invalid"
            escaped = False
            continue
        # The 8 legal simple escapes per RFC 8259:  "  \  /  b  f  n  r  t
        # (plus \uXXXX).
        #     '"a\\nb"'  -> \n is a legal escape -> fine
        #     '"a\\qb"'  -> \q is NOT legal      -> "invalid"

        if character == "\\":
            escaped = True
        elif character == '"':
            return index + 1          # <-- SUCCESS: index just PAST the close quote
        elif ord(character) < 0x20:
            return "invalid"
        # JSON forbids RAW control characters inside strings -- they must be
        # escaped. ord() gives the Unicode code point; < 0x20 is the C0 range
        # (which includes a literal newline 0x0A and tab 0x09).
        #
        # *** This is exactly what stops a string value from swallowing its own
        #     terminator: a raw \n inside an unclosed string is INVALID, so the
        #     only route to "complete" is close-quote THEN newline.

    return "prefix"
    # Ran off the end without a closing quote -> unfinished, but not wrong.

# WORKED EXAMPLES -------------------------------------------------------------
#   _json_string_end('"Beirut"',       0)  ->  8       (close quote at index 7)
#   _json_string_end('"Beirut',        0)  ->  "prefix"
#   _json_string_end('"Beirut"\n',     0)  ->  8       (caller inspects text[8:])
#   _json_string_end('"say \\"hi\\""', 0)  ->  11      (escaped quotes skipped)
#   _json_string_end('"Z\\u00fcrich"', 0)  ->  13      (\u + 4 hex consumed)
#   _json_string_end('"bad\\q"',       0)  ->  "invalid"
#   _json_string_end('Beirut"',        0)  ->  "invalid"   (no opening quote)
#
# FIDELITY NOTE ---------------------------------------------------------------
# It accepts ANY four hex digits after \u and does not validate surrogate
# pairing -- '"\ud800"' (a lone high surrogate) passes here. json.loads in
# [12] _parse_scalar is the backstop. Acceptable: this function's job is
# MASKING, not final validation.


# =============================================================================
# [8] scalar_prefix_status -- THE GRAMMAR ORACLE, called from inside [7]'s
#     hot loop. Dispatches to [9], [10] or [11].
# =============================================================================
def scalar_prefix_status(text: str, value_type: JsonType) -> PrefixStatus:
    """Classify a JSON scalar prefix whose required terminator is a newline."""
    # WHY A NEWLINE TERMINATOR IS MANDATORY FOR EVERY TYPE ---------------------
    # JSON scalars are NOT self-delimiting. Given "12" you cannot tell whether
    # the value is finished or whether a "3" is coming to make it 123. Same for
    # "tru" -> "true", or an open string. Requiring a \n gives the grammar an
    # unambiguous END SYMBOL, which is what makes the three-valued verdict
    # decidable at all.

    if value_type == "string":
        end = _json_string_end(text, 0) if text else "prefix"
        if isinstance(end, str):
            return end
        # !! CONVENTION: _json_string_end returns a UNION --
        #        int           on success (index just past the closing quote)
        #        PrefixStatus  on "not done" / "wrong"
        #    `isinstance(end, str)` is how the two are told apart. A poor-man's
        #    Result/Either type; Python has no built-in sum type, so
        #    union-plus-isinstance is the common workaround.
        #
        #    It works because int and str are disjoint -- but it is FRAGILE:
        #    a mistaken `if not end:` would misfire on position 0 AND on "".

        remaining = text[end:]
        if remaining == "\n":
            return "complete"
        return "prefix" if not remaining else "invalid"
        # EXAMPLES (text always starts with the forced quote from [7]):
        #   '"'          -> _json_string_end runs off the end   -> "prefix"
        #   '"Beir'      -> "prefix"
        #   '"Beirut'    -> "prefix"
        #   '"Beirut"'   -> end=8, remaining=''   -> "prefix"
        #                   (closed, awaiting \n or an <|im_end|> via Branch A)
        #   '"Beirut"\n' -> end=8, remaining='\n' -> "complete"
        #   '"Beirut",'  -> end=8, remaining=','  -> "invalid"
        #                   (the model tried to start building an object)
        #   '"Bei\nrut"' -> raw control char inside the string -> "invalid"

    if value_type == "boolean":
        return _literal_choices_status(text, ("true\n", "false\n"))
        # Only two legal strings -> no DFA needed.

    return _number_prefix_status(text, value_type == "integer")
    # *** `integer` and `number` SHARE one state machine, differentiated by a
    #     boolean flag. Avoids duplicating a 70-line DFA whose only difference
    #     is whether '.' and 'e' are reachable.


# =============================================================================
# [10] _literal_choices_status -- booleans
# =============================================================================
def _literal_choices_status(
    text: str, choices: tuple[str, ...]
) -> PrefixStatus:
    """Classify text against a finite set of exact literals."""
    if text in choices:
        return "complete"
    return "prefix" if any(choice.startswith(text) for choice in choices) else "invalid"
    # `any()` over a generator SHORT-CIRCUITS on the first hit.

# FULL BEHAVIOUR TABLE for choices == ("true\n", "false\n") -------------------
#   ''        -> "prefix"    (both literals start with the empty string)
#   't'       -> "prefix"
#   'tr'      -> "prefix"
#   'true'    -> "prefix"    <- CLOSED but not complete: still needs the \n
#   'true\n'  -> "complete"
#   'f'       -> "prefix"
#   'false\n' -> "complete"
#   'True'    -> "invalid"   <- *** PYTHON'S True IS CORRECTLY REJECTED ***
#   'yes'     -> "invalid"
#   '1'       -> "invalid"
#   'true '   -> "invalid"
#
# *** The 'True' case is a REAL trap in practice. A model trained on mountains
#     of Python source strongly wants to emit capital-T True for a boolean.
#     The mask makes that token unreachable and forces lowercase JSON `true`.
#     That is how "repeat_daily": true came out correct in the output file.
#
# NOTE the 'true' -> "prefix" row: after writing the word the decoder still
# needs either a \n token OR an <|im_end|>, which Branch A in [7] converts into
# a completion by testing 'true' + '\n'. Both paths work.


# =============================================================================
# [--] _literal_status -- DEAD (only the zero-parameter "{}\n" case used it)
# =============================================================================
def _literal_status(text: str, literal: str) -> PrefixStatus:
    """Classify text against one exact literal."""
    if text == literal:
        return "complete"
    return "prefix" if literal.startswith(text) else "invalid"
    # The single-choice twin of _literal_choices_status above.


# =============================================================================
# [11] _number_prefix_status -- integers AND numbers, one DFA
# =============================================================================
def _number_prefix_status(text: str, integer_only: bool) -> PrefixStatus:
    """Validate a JSON number prefix followed by exactly one newline."""
    # AN EXPLICIT FSM OVER THE JSON NUMBER GRAMMAR:
    #
    #   start --'-'-------> minus --+
    #     |                         |
    #     +--'0'------------------> +--> zero -----+
    #     +--'1'-'9'--------------> +--> integer --+ (loops on digits)
    #                                              |
    #      +------'.'-----> dot --digit--> fraction +--(loops)--+
    #      |                                                    |
    #      +------'e'/'E'-> exponent --'+'/'-'--> exponent_sign +
    #      |                    +--------digit------> exponent_digits
    #      +------'\n'----> DONE

    if not text:
        return "prefix"
    state = "start"
    for index, character in enumerate(text):
        if state == "start":
            if character == "-":
                state = "minus"
            elif character == "0":
                state = "zero"
            elif character in "123456789":
                state = "integer"
            # *** '0' and '1'-'9' go to DIFFERENT states. That separation is the
            #     ENTIRE mechanism for rejecting leading zeros.
            else:
                return "invalid"
                # EXAMPLES: '+5' invalid, '.5' invalid (JSON needs 0.5),
                #           ' 5' invalid, 'NaN' invalid, 'null' invalid.
        elif state == "minus":
            if character == "0":
                state = "zero"
            elif character in "123456789":
                state = "integer"
            else:
                return "invalid"
                # '-' alone stays in state 'minus' -> falls out as "prefix".
                # '-.'  -> invalid.   '-e' -> invalid.
        elif state in ("zero", "integer"):
            if state == "integer" and character.isdigit():
                continue
            # *** THE COMPACT-BUT-TRICKY LINE. The two states SHARE a branch
            #     because they accept the same TERMINATORS, but only `integer`
            #     may absorb more digits.
            #
            #     In state 'zero' a digit falls PAST this continue, then fails
            #     the \n / . / eE tests below, and lands on `return "invalid"`.
            #
            #     RESULT -- leading zeros rejected, as JSON requires:
            #         '0\n'   -> complete
            #         '0.5\n' -> complete   ('.' is still reachable from 'zero')
            #         '01\n'  -> INVALID    ***
            #         '007\n' -> INVALID    ***
            if character == "\n":
                state = "done"
            elif not integer_only and character == ".":
                state = "dot"
            elif not integer_only and character in "eE":
                state = "exponent"
            # *** `integer_only` GATES THE FLOAT SYNTAX. For an `integer`
            #     parameter, '.' and 'e' are literally unreachable states, so
            #     "3.7" can NEVER be produced where an int was declared.
            #     Enforced at the TOKEN-MASKING level, before any parsing.
            #
            #         type="integer":  '3' -> prefix,  '3.' -> INVALID
            #         type="number" :  '3' -> prefix,  '3.' -> prefix (state=dot)
            else:
                return "invalid"
        elif state == "dot":
            if character.isdigit():
                state = "fraction"
            else:
                return "invalid"
            # JSON requires AT LEAST ONE digit after the decimal point:
            #     '3.\n'  -> INVALID   (Python accepts 3. -- JSON does not)
            #     '3.5\n' -> complete
        elif state == "fraction":
            if character.isdigit():
                continue
            if character == "\n":
                state = "done"
            elif character in "eE":
                state = "exponent"
            else:
                return "invalid"
        elif state == "exponent":
            if character in "+-":
                state = "exponent_sign"
            elif character.isdigit():
                state = "exponent_digits"
            else:
                return "invalid"
        elif state == "exponent_sign":
            if character.isdigit():
                state = "exponent_digits"
            else:
                return "invalid"
            # '1e+\n' -> INVALID: an exponent needs at least one digit.
        elif state == "exponent_digits":
            if character.isdigit():
                continue
            if character == "\n":
                state = "done"
            else:
                return "invalid"
        else:
            return "invalid"
        if state == "done" and index != len(text) - 1:
            return "invalid"
        # *** ONCE THE \n IS CONSUMED, NOTHING MAY FOLLOW.
        #     Prevents '37.5\n and that is it' and '3\n\n'.
        #     `index != len(text) - 1` reads as "the \n was not the last char".
    return "complete" if state == "done" else "prefix"
    # Any half-formed but LEGAL state falls out as "prefix" so the decoder is
    # allowed to keep going: '-', '1.', '1e', '1e+', '3' are all "prefix".

# BEHAVIOUR TABLE -------------------------------------------------------------
#   text        integer_only=True   integer_only=False (number)
#   ---------   -----------------   ---------------------------
#   ''          prefix              prefix
#   '3'         prefix              prefix
#   '3\n'       complete            complete
#   '-'         prefix              prefix
#   '-42\n'     complete            complete
#   '0\n'       complete            complete
#   '01\n'      INVALID  ***        INVALID  ***
#   '3.'        INVALID  ***        prefix
#   '37.5\n'    INVALID  ***        complete
#   '3.\n'      INVALID             INVALID
#   '1e5\n'     INVALID             complete
#   '1e+\n'     INVALID             INVALID
#   '3\n '      INVALID             INVALID
#   'NaN\n'     INVALID             INVALID    (JSON has no NaN / Infinity)
#   '+3\n'      INVALID             INVALID
#
# !!!! ONE LATENT INCONSISTENCY -----------------------------------------------
# `character.isdigit()` is TRUE for NON-ASCII digits: Arabic-Indic U+0663,
# fullwidth U+FF13, superscript U+00B2.
#
# The `start` state uses a literal membership test `character in "123456789"`
# (correct, ASCII-only), but the CONTINUATION states use .isdigit(). So a
# string like "1" followed by U+0663 then "\n":
#
#     _number_prefix_status(...)  ->  "complete"        (!!)
#     json.loads(...)             ->  JSONDecodeError
#
# The token would be ACCEPTED by the mask, then blow up in [12] _parse_scalar
# as a DecodingError -- instead of being silently skipped in favour of the next
# token, which is what SHOULD happen.
#
# Practically unreachable (the model would have to rank such a token at the
# top), but swapping .isdigit() for `character in "0123456789"` closes it for
# free. Applies to the 'integer', 'dot', 'fraction', 'exponent',
# 'exponent_sign' and 'exponent_digits' branches above.


# #############################################################################
# #  APPENDIX A -- THREE EXAMPLES END TO END                                  #
# #############################################################################
#
# === EXAMPLE 1: "Set an alarm for 06:30 and repeat it every day" =============
#
# PHASE 1  choose_function
#     trie -> "fn_set_alarm"      (3 forward passes)
#
# PHASE 2a  time: string
#     fixed_prefix = '"'  -> prompt ends:  ...JSON value for time:"
#     '"0'       prefix
#     '"06'      prefix
#     '"06:'     prefix
#     '"06:3'    prefix
#     '"06:30'   prefix
#     '"06:30"'  prefix          <- closed, awaiting terminator
#     <|im_end|>                 <- BRANCH A: '"06:30"' + '\n' -> complete
#     -> "06:30"   CORRECT
#
# PHASE 2b  repeat_daily: boolean
#     fixed_prefix = ''   (no forced quote for booleans)
#     rank 0: 'Yes'   -> _literal_choices_status -> invalid -> SKIP   ***
#     rank 1: 'True'  -> invalid                           -> SKIP   ***
#     rank 2: 'true'  -> prefix -> COMMIT
#     <|im_end|>      -> 'true' + '\n' -> complete
#     -> True   CORRECT
#
#     *** The mask silently corrected the model TWICE here.
#
# RESULT (matches data/output/function_calling_results.json exactly):
#     {"prompt": "...", "name": "fn_set_alarm",
#      "parameters": {"time": "06:30", "repeat_daily": true}}
#
#
# === EXAMPLE 2: "Convert 37.5 degrees to Fahrenheit" =========================
#
# PHASE 2a  value: number       -> integer_only = False, so '.' is reachable
#     '2'      prefix   (state=integer)
#     '22'     prefix
#     '22.'    prefix   (state=dot)     <- would be INVALID for type "integer"
#     '22.5'   prefix   (state=fraction)
#     <|im_end|> -> '22.5\n' -> complete
#     -> 22.5
#
# !! THE PROMPT SAID 37.5. The model produced 22.5.
#    22.5 is a PERFECTLY LEGAL JSON number, so the grammar has no objection.
#    THE MASK GUARANTEES TYPE CORRECTNESS, NOT SEMANTIC CORRECTNESS.
#    A Qwen3-0.6B capability limit -- fixable with better prompting or a larger
#    model, NEVER with a tighter grammar.
#
# PHASE 2b  target_scale: string -> "target_scale"  (echoed the KEY NAME --
#           same class of model failure, again perfectly type-valid)
#
#
# === EXAMPLE 3: a failure path ===============================================
#
# Suppose `passengers: integer` and the model is dead set on prose.
#     iter 0: walks all ~151k ranked tokens; every candidate is invalid
#             (' ', 'Three', 'three', '"', '{', 'null', ...)
#             -> the for loop NEVER breaks -> the for/else fires
#             -> DecodingError("no type-valid token for parameter 'passengers'")
#
# main() catches it:
#     except (InputFileError, DecodingError) as error:
#         print(f"error: {error}", file=sys.stderr)
#         return 1
#
#     $ error: no type-valid token for parameter 'passengers'
#     $ echo $?   ->   1
#
# In reality this never fires for numbers: SOME digit token always exists
# somewhere in the ranking. The realistic failure is the OTHER one --
# "parameter 'origin' exceeded 96 tokens" -- a string that never closes.
#
#
# #############################################################################
# #  APPENDIX B -- PYTHON CONVENTIONS CHEAT SHEET                             #
# #############################################################################
#
#  CONSTRUCT                      THE THING THAT TRIPS PEOPLE UP
#  -----------------------------  --------------------------------------------
#  from __future__ import         Annotations become strings; enables
#      annotations                `int | PrefixStatus` pre-3.10, kills import
#                                 cycles. Free.
#
#  Literal["a","b"]               Type-checker-only. Values are plain `str` at
#                                 runtime -- hence `== "complete"`, not `is`.
#
#  cast(T, x)                     Returns `x` UNCHANGED. Validates NOTHING.
#                                 Purely a note to mypy.
#
#  for ... else                   `else` runs ONLY IF NO `break`. Read it as
#                                 "NO-BREAK". `return` also skips it;
#                                 `continue` does NOT.
#
#  int | PrefixStatus  return     Poor-man's Result type, disambiguated by
#                                 `isinstance(x, str)`. Fragile -- never test
#                                 it with truthiness.
#
#  isinstance(True, int) is True  *** bool SUBCLASSES int. *** This is why
#                                 `not isinstance(value, bool)` is mandatory,
#                                 not decorative.
#
#  next(genexpr)                  "Find first match." Raises StopIteration if
#                                 empty (cannot happen here).
#
#  raise X from error             Chains __cause__; traceback shows BOTH
#                                 exceptions. `from None` would suppress.
#
#  f"{value!r}"                   `!r` = repr() -> value appears quoted,
#                                 whitespace visible.
#
#  logits.__getitem__ as key=     Bound method as key function -- no Python
#                                 stack frame per comparison.
#
#  list(some_dict)                Yields KEYS, not items, not values.
#
#  dict insertion order           Guaranteed since 3.7; Pydantic preserves JSON
#                                 key order -> deterministic parameter order.
#
#  .isdigit() != ASCII            True for U+0663, U+FF13, U+00B2. The latent
#                                 hole flagged in [11].
#
#
# #############################################################################
# #  APPENDIX C -- CALL FLOW                                                  #
# #############################################################################
#
#   __main__.main()
#     |
#     +- load_model() x2 ---------> Pydantic validates catalog + prompts
#     +- Small_LLM_Model()
#     +- pipeline.generate_calls()
#          |
#          +- FOR EACH PROMPT ------------------------------------------+
#               |                                                       |
#     +=========v==============================================+        |
#     | PHASE 1 -- decoding.choose_function()            [1]   |        |
#     +========================================================+        |
#     |  +- len==1? --> return immediately (no inference)      |        |
#     |  +- _chat_prompt()   ChatML + empty <think></think> [2]|        |
#     |  +- _encode()        unwrap the [1,N] tensor       [3] |        |
#     |  +- _encode_choice() xN  build the token trie      [4] |        |
#     |  +- _decode_choice()  <== TRIE-CONSTRAINED LOOP    [5] |        |
#     |       while active:                                    |        |
#     |         any branch finished? --> return it             |        |
#     |         allowed = {next id of each live branch}        |        |
#     |         get_logits_from_input_ids(prefix)              |        |
#     |         argmax over `allowed` ONLY  <== THE MASK       |        |
#     |         append + prune non-matching branches           |        |
#     +=========+==============================================+        |
#               | FunctionDefinition                                    |
#     +=========v==============================================+        |
#     | PHASE 2 -- decoding.generate_parameters()        [6]   |        |
#     +========================================================+        |
#     |  FOR EACH PARAMETER (in declaration order)             |        |
#     |   +- _generate_scalar()  <== GRAMMAR-CONSTRAINED   [7] |        |
#     |       fixed_prefix = '"' if string else ''             |        |
#     |       _chat_prompt() -> _encode()                      |        |
#     |       repeat <= 96 times:                              |        |
#     |         get_logits_from_input_ids(prefix)              |        |
#     |         ranked = argsort(logits, desc)   # ~151k       |        |
#     |         for token in ranked:                           |        |
#     |           candidate = decode(generated + [token])      |        |
#     |           +--------------------------------------+     |        |
#     |           | BRANCH A: no new text (EOS/special)  |     |        |
#     |           |   complete-if-terminated? --> DONE   |     |        |
#     |           |   else: VETO the stop, next token    |     |        |
#     |           +--------------------------------------+     |        |
#     |           scalar_prefix_status(candidate, type)  [8]   |        |
#     |             +- "invalid"  --> skip  <== THE MASK       |        |
#     |             +- "prefix"   --> commit, next position    |        |
#     |             +- "complete" --> _parse_scalar()    [12]  |        |
#     |         else: DecodingError (whole vocab rejected)     |        |
#     +=========+==============================================+        |
#               | dict[str, Any]                                        |
#               +- pipeline._validate_parameters()  (3rd type check)    |
#               +- append FunctionCallResult --------------------------->+
#                             |
#              write_model() -+-> data/output/function_calling_results.json
#
#
#   scalar_prefix_status [8] dispatch:
#         +- "string"   -> _json_string_end()          [9]  RFC 8259 scanner
#         +- "boolean"  -> _literal_choices_status()  [10]  {"true\n","false\n"}
#         +- int/number -> _number_prefix_status()    [11]  9-state DFA
#                                                           (integer_only gates
#                                                            '.' and 'e')
#
#   UNREACHABLE (object-level grammar, superseded):
#         object_prefix_status --+- _consume_literal
#                                +- _consume_scalar -> _json_string_end
#         _literal_status  (only the zero-parameter "{}\n" case)
#
#
#   THREE INDEPENDENT LAYERS OF VALIDATION ON THE SAME DATA:
#         1. the grammar         (masking, during generation)
#         2. _parse_scalar       (json.loads + exact type check)
#         3. _validate_parameters (pipeline.py: key set + type check again)
#
#   Unusual, but coherent for a system whose premise is "output is PROVABLY
#   schema-compliant". Each layer is an independent proof; the redundancy is
#   the point, not an oversight.
