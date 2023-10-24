
def test_ast():
    from stanza import func

    def f():
        pass
    f_wrapped = func(f)
    print(f_wrapped)
    assert False