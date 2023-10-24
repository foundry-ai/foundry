def f():
    pass

import stanza
from stanza import func

a, b = (lambda x: x), (lambda y: y + 1)
print(stanza.parse(a))
print(stanza.parse(b))

f_ast = stanza.parse(f)
print(f_ast)
# # Also works on lambdas!
# f_func = stanza.func(f)

# # Testing transpiling mutually recursive functions:
# @func
# def foo():
#     return bar()

# @func
# def bar():
#     return foo()