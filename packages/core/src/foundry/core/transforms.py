import jax
import jax.tree_util
import functools
import weakref

def vmap(func, /, in_axes=0, out_axes=0, *, axis_name=None):
    return jax.vmap(func, 
        in_axes=in_axes, 
        out_axes=out_axes, axis_name=None
    )

# A filtered jit, which makes anything
# that is not a jax Array type a static argument.
def jit(func=None, *, 
            allow_static=False, allow_arraylike=False,
            donate_argnums=None, donate_argnames=None
        ):
    if func is None:
        return functools.partial(jit,
            allow_static=allow_static,
            allow_arraylike=allow_arraylike,
            donate_argnums=donate_argnums,
            donate_argnames=donate_argnames
        )
    func = _make_filtered(func)
    func = jax.jit(func, 
        donate_argnums=donate_argnames,
        donate_argnames=donate_argnames
    )
    # if static arguments are allowed, return a wrapper
    # function which auto-filters static arguments
    # and wraps them with a _Static "bubble"
    # Otherwise, ensure that there are only array arguments.
    if allow_static: 
        func = _make_filtering(func)
    else: 
        func = _ensure_array_args(func)
    # if allow_arraylike is true
    if allow_arraylike: 
        func = _convert_arraylike_args(func)

    return func

# Functions 
def _is_array(x):
    return isinstance(x, jax.Array)

class _Static:
    def __init__(self, value):
        self.value = value
    def __repr__(self):
        return self.value.__repr__()
    def __str__(self):
        return self.value.__str__()
    def __hash__(self):
        return hash(self.value)
    def __eq__(self, other):
        return self.value == other

jax.tree_util.register_static(_Static)

_FILTERED_FUNCS = weakref.WeakKeyDictionary()

def _make_filtered(func):
    def unwrap(x):
        # map _Static types 
        # back to their original values
        # everything else must be an array
        if isinstance(x, _Static):
            return x.value
        elif not _is_array(x):
            raise ValueError(
                "Arguments to filtered functions"
                " must be either static variables or jax arrays"
            )
        return x
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        args, kwargs = jax.tree_util.tree_map(
            unwrap, (args, kwargs), 
            is_leaf=lambda x: isinstance(x, _Static)
        )
        return func(*args, **kwargs)
    if func not in _FILTERED_FUNCS:
        _FILTERED_FUNCS[func] = wrapper
    return _FILTERED_FUNCS[func]
    
_FILTERING_FUNCS = weakref.WeakKeyDictionary()

def _make_filtering(func):
    def wrap(x):
        if not isinstance(x, jax.Array):
            return _Static(x)
        return x
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        args, kwargs = jax.tree_util.tree_map(wrap, (args, kwargs))
        return func(*args, **kwargs)
    if func not in _WRAPPING_FUNCS:
        _WRAPPING_FUNCS[func] = wrapper
    return _WRAPPING_FUNCS[func]

_ARRAY_ARG_FUNCS = weakref.WeakKeyDictionary()

def _assert_array_args(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        leaves = jax.tree_util.tree_leaves((args, kwargs))
        if any(not _is_array(x) for x in leaves):
            raise ValueError("Cannot pass non-array argument to jit'd function without allow_static=True!")
        return func(*args, **kwargs)
    if func not in _ARRAY_ARG_FUNCS:
        _ARRAY_ARG_FUNCS[func] = wrapper
    return _ARRAY_ARG_FUNCS[func]

_CONVERT_ARRAYLIKE_FUNCS = weakref.WeakKeyDictionary()

def _convert_arraylike_args(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        args, kwargs = jax.tree_util.tree_map(lambda x: jnp.array(x), (args, kwargs))
        return func(*args, **kwargs)
    if func not in _CONVERT_ARRAYLIKE_FUNCS:
        _CONVERT_ARRAYLIKE_FUNCS[func] = wrapper
    return _CONVERT_ARRAYLIKE_FUNCS[func]