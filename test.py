"""Vocabulary loading and token classification for constrained decoding.

Loads the model vocabulary once at startup and pre-classifies every token
into categories needed by the constrained decoder. All lookups during
generation are O(1) set membership checks.
"""

import json
import sys
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Optional, Set

from src.models import JsonSchemaType, TokenId, TokenStr, VocabMapping


# ---------------------------------------------------------------------------
# Qwen3 / BPE token decoding helpers
# ---------------------------------------------------------------------------

# BPE vocabularies encode certain byte values as Unicode look-alikes so that
# every byte is representable as a printable character. Qwen3 uses the same
# mapping as GPT-2. We need to reverse it to get the actual string a token
# represents.

def _build_byte_decoder() -> Dict[str, str]:
    """Build a mapping from unicode escapes back to real chars."""
    # we are creating a list of ascii code of all printable characters
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )

    # we make a copy
    cs = bs[:]

    n = 0

    for b in range(256):
        # if the char is not printable
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1

    return {chr(b): chr(c) for b, c in zip(cs, bs)}


_BYTE_DECODER: Dict[str, str] = _build_byte_decoder()


def decode_bpe_token(token: str) -> str:
    """This function is responsible for converting one token from the vocabulary into the actual text it represents."""

    # some tokens are special control tokens, not normal text
    # per example: <|user|>
    if token.startswith("<|") and token.endswith("|>"):
        return token

    try:
        return "".join(_BYTE_DECODER.get(c, c) for c in token)
    except Exception:
        return token


# ---------------------------------------------------------------------------
# Token classification sets
# ---------------------------------------------------------------------------

@dataclass
class TokenSets:
    """Pre-classified sets of token IDs for each JSON structural role.
    """

    # frozenset is used because these token groups are built once and should never change afterward

    open_brace: FrozenSet[TokenId] = field(default_factory=frozenset)
    close_brace: FrozenSet[TokenId] = field(default_factory=frozenset)
    open_bracket: FrozenSet[TokenId] = field(default_factory=frozenset)
    close_bracket: FrozenSet[TokenId] = field(default_factory=frozenset)
    quote: FrozenSet[TokenId] = field(default_factory=frozenset)
    colon: FrozenSet[TokenId] = field(default_factory=frozenset)
    comma: FrozenSet[TokenId] = field(default_factory=frozenset)

    whitespace: FrozenSet[TokenId] = field(default_factory=frozenset)

    # tokens that start a number literal ('0-9', '-')
    number_start: FrozenSet[TokenId] = field(default_factory=frozenset)

    number_continue: FrozenSet[TokenId] = field(default_factory=frozenset)

    string_interior: FrozenSet[TokenId] = field(default_factory=frozenset)

    # tokens that are exactly 't', 'r', 'u', 'e' (for true literal)
    true_chars: FrozenSet[TokenId] = field(default_factory=frozenset)

    # tokens that are exactly 'f', 'a', 'l', 's', 'e' (for false literal)
    false_chars: FrozenSet[TokenId] = field(default_factory=frozenset)

    # tokens that are exactly 'n', 'u', 'l' (for null literal)
    null_chars: FrozenSet[TokenId] = field(default_factory=frozenset)

    # tokens that decode to a complete keyword or prefix of one
    # e.g. "true", "false", "null" as single tokens
    keyword_tokens: FrozenSet[TokenId] = field(default_factory=frozenset)

    # all token IDs whose decoded text is non-empty
    non_empty: FrozenSet[TokenId] = field(default_factory=frozenset)


# ---------------------------------------------------------------------------
# VocabularyAnalyzer
# ---------------------------------------------------------------------------

class VocabularyAnalyzer:
    """Loads the model vocabulary and provides token classification."""

    def __init__(self, vocab_path: str) -> None:
        """Load vocabulary and pre-classify all tokens."""
        self.vocab_path = vocab_path
        self.token_to_id: Dict[TokenStr, TokenId] = {}
        self.id_to_decoded: Dict[TokenId, str] = {}
        self.sets = TokenSets()

        self.load(vocab_path)
        self.classify()

    def load(self, path: str) -> None:
        """Load and parse the vocabulary JSON file.

        The vocabulary JSON maps token strings to integer IDs:
            { "!": 0, "\"": 1, ... }

        Args:
            path: Path to the vocab JSON file.

        Raises:
            SystemExit: If the file cannot be read or parsed.
        """
        try:
            with open(path, "r", encoding="utf-8") as fh:
                raw: Dict[str, int] = json.load(fh)
        except FileNotFoundError:
            print(
                f"[ERROR] Vocabulary file not found: '{path}'",
                file=sys.stderr,
            )
            sys.exit(1)
        except json.JSONDecodeError as exc:
            print(
                f"[ERROR] Vocabulary file contains invalid JSON: {exc}",
                file=sys.stderr,
            )
            sys.exit(1)
        except OSError as exc:
            print(
                f"[ERROR] Could not read vocabulary file: {exc}",
                file=sys.stderr,
            )
            sys.exit(1)

        self._token_to_id = raw
        vocab_size = len(raw)

        # Build id → decoded string mapping
        for token_str, token_id in raw.items():
            decoded = decode_bpe_token(token_str)
            self._id_to_decoded[token_id] = decoded

        print(f"[INFO] Loaded vocabulary: {vocab_size} tokens.")

    def _classify(self) -> None:
        """Pre-classify all tokens into structural categories.

        Iterates the entire vocabulary once, assigning each token to the
        appropriate set(s) based on its decoded string value.
        """
        open_brace: Set[TokenId] = set()
        close_brace: Set[TokenId] = set()
        open_bracket: Set[TokenId] = set()
        close_bracket: Set[TokenId] = set()
        quote: Set[TokenId] = set()
        colon: Set[TokenId] = set()
        comma: Set[TokenId] = set()
        whitespace: Set[TokenId] = set()
        number_start: Set[TokenId] = set()
        number_continue: Set[TokenId] = set()
        string_interior: Set[TokenId] = set()
        true_chars: Set[TokenId] = set()
        false_chars: Set[TokenId] = set()
        null_chars: Set[TokenId] = set()
        keyword_tokens: Set[TokenId] = set()
        non_empty: Set[TokenId] = set()

        _number_start_chars = set("0123456789-")
        _number_continue_chars = set("0123456789-.eE+")
        _true_chars = set("true")
        _false_chars = set("false")
        _null_chars = set("nul")
        _keywords = {"true", "false", "null"}

        for token_id, decoded in self._id_to_decoded.items():
            if not decoded:
                continue

            non_empty.add(token_id)

            # --- Structural single-char tokens ---
            if decoded == "{":
                open_brace.add(token_id)
            if decoded == "}":
                close_brace.add(token_id)
            if decoded == "[":
                open_bracket.add(token_id)
            if decoded == "]":
                close_bracket.add(token_id)
            if decoded == '"':
                quote.add(token_id)
            if decoded == ":":
                colon.add(token_id)
            if decoded == ",":
                comma.add(token_id)

            # --- Whitespace ---
            if decoded.strip() == "" and decoded != "":
                whitespace.add(token_id)

            # --- Number tokens ---
            if decoded[0] in _number_start_chars:
                # All chars must be valid number characters
                if all(c in _number_continue_chars for c in decoded):
                    number_start.add(token_id)
                    number_continue.add(token_id)
            elif all(c in _number_continue_chars for c in decoded):
                number_continue.add(token_id)

            # --- String interior tokens ---
            # Valid inside a JSON string: no raw unescaped " or \n or \r
            # (backslash sequences are OK, but raw control chars are not)
            if self._is_valid_string_interior(decoded):
                string_interior.add(token_id)

            # --- Boolean/null character tokens ---
            if len(decoded) == 1:
                if decoded in _true_chars:
                    true_chars.add(token_id)
                if decoded in _false_chars:
                    false_chars.add(token_id)
                if decoded in _null_chars:
                    null_chars.add(token_id)

            # --- Complete keyword tokens (true/false/null as single token) ---
            stripped = decoded.strip()
            if stripped in _keywords:
                keyword_tokens.add(token_id)

        # Freeze all sets for thread safety and hashability
        self.sets = TokenSets(
            open_brace=frozenset(open_brace),
            close_brace=frozenset(close_brace),
            open_bracket=frozenset(open_bracket),
            close_bracket=frozenset(close_bracket),
            quote=frozenset(quote),
            colon=frozenset(colon),
            comma=frozenset(comma),
            whitespace=frozenset(whitespace),
            number_start=frozenset(number_start),
            number_continue=frozenset(number_continue),
            string_interior=frozenset(string_interior),
            true_chars=frozenset(true_chars),
            false_chars=frozenset(false_chars),
            null_chars=frozenset(null_chars),
            keyword_tokens=frozenset(keyword_tokens),
            non_empty=frozenset(non_empty),
        )

        print(
            f"[INFO] Token classification complete: "
            f"{len(open_brace)} '{{', "
            f"{len(quote)} '\"', "
            f"{len(number_start)} number-start, "
            f"{len(string_interior)} string-interior tokens."
        )

    @staticmethod
    def _is_valid_string_interior(text: str) -> bool:
        """Check whether text is safe to emit inside a JSON string literal.

        A token is valid inside a JSON string if it contains no raw
        unescaped double-quote (") and no raw unescaped control characters
        (\\x00-\\x1f except when they appear as part of an escape sequence).

        We allow backslash itself — the constrained decoder will track
        escape context separately if needed.

        Args:
            text: Decoded token string to check.

        Returns:
            True if the token can appear inside a JSON string value.
        """
        for ch in text:
            # Raw unescaped quote would end the string prematurely
            if ch == '"':
                return False
            # Raw control characters (except tab which is \\t) are illegal
            # in JSON strings unless escaped
            code = ord(ch)
            if code < 0x20 and ch not in ("\t",):
                return False
        return True

    # ---------------------------------------------------------------------------
    # Public accessors
    # ---------------------------------------------------------------------------

    def decode_token(self, token_id: TokenId) -> str:
        """Return the decoded string for a token ID.

        Args:
            token_id: Integer token ID.

        Returns:
            Decoded string, or empty string if ID is unknown.
        """
        return self._id_to_decoded.get(token_id, "")

    def token_id_for(self, token_str: str) -> Optional[TokenId]:
        """Return the token ID for a raw (BPE-encoded) token string.

        Args:
            token_str: Raw token string as it appears in the vocab file.

        Returns:
            Integer token ID, or None if not found.
        """
        return self._token_to_id.get(token_str)

    def get_vocab_mapping(self) -> VocabMapping:
        """Return a copy of the full id → decoded-string mapping.

        Returns:
            Dict mapping token ID to decoded string.
        """
        return dict(self._id_to_decoded)

    def vocab_size(self) -> int:
        """Return the total number of tokens in the vocabulary.

        Returns:
            Number of entries in the vocabulary.
        """
        return len(self._id_to_decoded)

    def tokens_for_type(self, schema_type: JsonSchemaType) -> FrozenSet[TokenId]:
        """Return the set of token IDs that can START a value of the given type.

        This is the primary interface for the constrained decoder to ask
        "what tokens can begin a value of this schema type?".

        Args:
            schema_type: JSON schema type string.

        Returns:
            Frozen set of valid starting token IDs for this type.
        """
        if schema_type in ("number", "integer"):
            return self.sets.number_start
        if schema_type == "string":
            return self.sets.quote
        if schema_type == "boolean":
            return self.sets.keyword_tokens
        if schema_type == "array":
            return self.sets.open_bracket
        if schema_type == "object":
            return self.sets.open_brace
        # Fallback: allow any non-empty token
        return self.sets.non_empty
