from jax.tree_util import Partial

# Will wrap a function into a jax-vmappable type
def fun(f):
    if isinstance(f, Partial):
        return f
    else:
        return Partial(f)