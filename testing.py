def build_byte_decoder():
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

# print(build_byte_decoder())

print(chr(22))

print(chr(257))