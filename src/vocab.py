from typing import Dict, FrozenSet, Set, Optional
from dataclasses import dataclass, field
import json
import sys

from src.models import JsonSchemaType, TokenId, TokenStr, VocabMapping


def build_byte_decoder() -> Dict[str, str]:
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


BYTE_DECODER: Dict[str, str] = build_byte_decoder()


def decode_bpe_token(token: str) -> str:
    """This function is responsible for converting one token from the vocabulary into the actual text it represents."""

    # some tokens are special control tokens, not normal text
    # per example: <|user|>
    if token.startswith("<|") and token.endswith("|>"):
        return token

    try:
        return "".join(BYTE_DECODER.get(c, c) for c in token)
    except Exception:
        return token


@dataclass
class TokenSets:
    """Pre-classified sets of token IDs for each JSON structural role."""

    # frozenset is used because these token groups are built once and should never change afterward
    # field is used to tell the dataclass how to create the default value for each field

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
        """Load and parse the vocabulary JSON file."""
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

        self.token_to_id = raw
        vocab_size = len(raw)

        for token_str, token_id in raw.items():
            decoded = decode_bpe_token(token_str)
            self.id_to_decoded[token_id] = decoded

        print(f"[INFO] Loaded vocabulary: {vocab_size} tokens.")

    def classify(self) -> None:
        """Pre-classify all tokens into structural categories."""
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

        number_start_chars = set("0123456789-")
        number_continue_chars = set("0123456789-.eE+")
        true_chars = set("true")
        false_chars = set("false")
        null_chars = set("nul")
        keywords = {"true", "false", "null"}

        for token_id, decoded in self.id_to_decoded.items():
            if not decoded:
                continue

            non_empty.add(token_id)

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

            if decoded.strip() == "" and decoded != "":
                whitespace.add(token_id)

            if decoded[0] in number_start_chars:
                if all(c in number_continue_chars for c in decoded):
                    number_start.add(token_id)
                    number_continue.add(token_id)
            elif all(c in number_continue_chars for c in decoded):
                number_continue.add(token_id)

            if self.is_valid_string_interior(decoded):
                string_interior.add(token_id)

            if len(decoded) == 1:
                if decoded in true_chars:
                    true_chars.add(token_id)
                if decoded in false_chars:
                    false_chars.add(token_id)
                if decoded in null_chars:
                    null_chars.add(token_id)

            stripped = decoded.strip()
            if stripped in keywords:
                keyword_tokens.add(token_id)

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
    def is_valid_string_interior(text: str) -> bool:
        """Check whether text is safe to emit inside a JSON string literal."""
        for ch in text:
            if ch == '"':
                return False
            code = ord(ch)
            if code < 32 and ch not in ("\t",):
                return False
        return True


    def decode_token(self, token_id: TokenId) -> str:
        """Return the decoded string for a token ID."""
        return self.id_to_decoded.get(token_id, "")

    def token_id_for(self, token_str: str) -> Optional[TokenId]:
        """Return the token ID for a raw (BPE-encoded) token string."""
        return self.token_to_id.get(token_str)

    def get_vocab_mapping(self) -> VocabMapping:
        """Return a copy of the full id to decoded-string mapping."""
        return dict(self.id_to_decoded)

    def vocab_size(self) -> int:
        """Return the total number of tokens in the vocabulary."""
        return len(self.id_to_decoded)

    def tokens_for_type(self, schema_type: JsonSchemaType) -> FrozenSet[TokenId]:
        """Return the set of token IDs that can START a value of the given type."""
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
        
        return self.sets.non_empty