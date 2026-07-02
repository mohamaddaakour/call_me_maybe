from typing import Dict, FrozenSet
from dataclasses import dataclass, field

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
    """Pre-classified sets of token IDs for each JSON structural role.
    """

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
