

def read_chars(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Get unique characters
    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    return text, chars, vocab_size


def create_vocab_dicts(chars):
    # create a mapping of characters to integers
    stoi = {ch: i for i, ch in enumerate(chars)}
    itoi = {i: ch for i, ch in enumerate(chars)}
    return stoi, itoi


def create_encode(stoi):
    def encode(s): return [stoi[c] for c in s]
    return encode

def create_decode(atoi):
    def decode(l): return ''.join([itoi[i] for i in l])
    return decode