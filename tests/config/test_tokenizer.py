from stanza.runtime.config import ArgTokenizer

def test_tokenizer():
    tokenizer = ArgTokenizer(["foo", "--bar=baz", "--bar", "baz", "-v=1", "-a=2"])
    print(list(tokenizer))
    assert False