import sys
import inspect
import jax.tree_util
import types

from functools import partial
from typing import (
    Dict, Tuple, NamedTuple, TypeVar, 
    Callable, overload
)

if sys.version_info >= (3, 11):
    from typing import dataclass_transform
else:
    from typing_extensions import dataclass_transform

class _MISSING_TYPE:
    def __repr__(self):
        return "Missing"

MISSING = _MISSING_TYPE()

class Field:
    """A struct field class. Not to be instantiated directly. Use the field() function instead.
    """
    __slots__ = (
        'name',
        'type',
        'pytree_node',
        'kw_only',
        'default',
        'default_factory',
        # unlike default_factory, initializer
        # is called with a partially-initialized
        # instance and can be used to set fields
        # based on earlier fields. Either default, default_factory, or initializer
        # must be specified
        'initializer'
    )
    def __init__(self, *, name, type, 
                 pytree_node=True, kw_only=False,
                 default=MISSING, default_factory=MISSING, initializer=MISSING):
        self.name = name
        self.type = type
        self.pytree_node = pytree_node
        self.kw_only = kw_only
        self.default = default
        self.default_factory = default_factory
        self.initializer = initializer

    # if this is a required field
    @property
    def required(self):
        return self.default is MISSING and self.default_factory is MISSING and self.initializer is MISSING
    
    def __repr__(self):
        return (f"Field(name={self.name}, type={self.type}, "
            f"pytree_node={self.pytree_node}, default={self.default}, "
             f"default_factory={self.default_factory}, initializer={self.initializer})")

def field(*, pytree_node=True, kw_only=False, 
          default=MISSING, default_factory=MISSING,
          initializer=MISSING):
    return Field(name=None, type=None,
        pytree_node=pytree_node,
        kw_only=kw_only,
        default=default,
        default_factory=default_factory,
        initializer=initializer
    )
    
def fields(struct):
    return struct.__struct_fields__.values()

def replace(_struct, **kwargs):
    cls = _struct.__class__
    s = cls.__new__(cls)
    for k in _struct.__struct_fields__.keys():
        v = getattr(_struct, k)
        if k in kwargs: v = kwargs[k]
        object.__setattr__(s, k, v)
    return s

class StructParams(NamedTuple):
    kw_only: bool

_C = TypeVar("_C", bound=type)

@overload
@dataclass_transform(field_specifiers=(field,),
    frozen_default=True)
def dataclass(cls : None = ..., *,
    kw_only : bool = ...) -> Callable[[_C], _C]: ...

@overload
@dataclass_transform(field_specifiers=(field,),
    frozen_default=True)
def dataclass(cls : _C, *, kw_only : bool = ...) -> _C: ...

# actual dataclass implementation
def dataclass(maybe_cls=None, *, kw_only=False) -> _C:
    params = StructParams(kw_only=kw_only)
    if maybe_cls is None:
        return partial(make_dataclass, params=params)
    return make_dataclass(maybe_cls, params)

def make_dataclass(cls, params):
    fields, pos_fields, kw_fields = _collect_fields(cls, params)

    if cls.__module__ in sys.modules:
        globals = sys.modules[cls.__module__].__dict__
    else:
        globals = {}
    # create a new class that extends base_struct and has Struct type mixing
    cls.__struct_params__ = params
    cls.__struct_fields__ = fields
    cls.__slots__ = tuple(field.name for field in fields.values())
    cls.__init__ = _make_init(cls, fields, pos_fields, kw_fields, globals)
    cls.__setattr__ = _make_frozen_setattr(cls)
    cls.__repr__ = lambda self: f"{cls.__name__}({', '.join(f'{k}={getattr(self, k)!r}' for k in fields)})"

    if not getattr(cls, '__doc__'):
        # Create a class doc-string, same a dataclasses
        try:
            text_sig = str(inspect.signature(cls)).replace(' -> None', '')
        except (TypeError, ValueError):
            text_sig = ''
        cls.__doc__ = (cls.__name__ + text_sig)
    _register_jax_type(cls)
    return cls

def _register_jax_type(cls):
    from jax.tree_util import GetAttrKey
    dyn_fields = list(f for f in fields(cls) if f.pytree_node)
    static_fields = list(f for f in fields(cls) if not f.pytree_node)
    def flatten(v):
        children = tuple(getattr(v, f.name) for f in dyn_fields)
        # children = tuple(FnChild(v) if type(v) == types.FunctionType else v for v in children)
        static_children = tuple(getattr(v, f.name) for f in static_fields)
        return children, static_children
    def flatten_with_keys(v):
        children = tuple((GetAttrKey(f.name),getattr(v, f.name)) for f in dyn_fields)
        # children = tuple(FnChild(v) if type(v) == types.FunctionType else v for v in children)
        static_children = tuple(getattr(v, f.name) for f in static_fields)
        return children, static_children
    def unflatten(static_children, children):
        i = cls.__new__(cls)
        # unwrap FnNode
        # children = tuple(c.fn if type(c) == FnChild else c for c in children)
        for f, c in zip(dyn_fields, children):
            object.__setattr__(i, f.name, c)
        for f, c in zip(static_fields, static_children):
            object.__setattr__(i, f.name, c)
        return i
    jax.tree_util.register_pytree_with_keys(
        cls, flatten_with_keys, unflatten, flatten
    )

def _collect_fields(cls, params) -> Tuple[Dict[str, Field], Dict[str, Field], Dict[str, Field]]:
    fields = {} # type: Dict[str, Field]
    # handle inheritance
    for b in cls.__mro__[-1:0:-1]:
        sub_fields = getattr(b, "__struct_fields__", None)
        if sub_fields is not None:
            # sub_params = b.__struct_params__
            for f in sub_fields.values():
                fields[f.name] = f

    annotations = inspect.get_annotations(cls)
    annotation_fields = {}
    for name, _type in annotations.items():
        f = getattr(cls, name, MISSING)
        if not isinstance(f, Field):
            f = Field(name=None, type=None, kw_only=params.kw_only, default=f)
        # re-instantiate the field with the name and type
        f = Field(name=name, type=_type, pytree_node=f.pytree_node,
                  kw_only=f.kw_only or params.kw_only, default=f.default,
            default_factory=f.default_factory, initializer=f.initializer)
        annotation_fields[name] = f

    for name, value in cls.__dict__.items():
        if isinstance(value, Field) and not name in annotation_fields:
            raise TypeError(f"field {name} has no type annotation")

    # add fields from annotations and delete them from the class if default
    for f in annotation_fields.values():
        fields[f.name] = f
        if f.default is MISSING and hasattr(cls, f.name):
            delattr(cls, f.name)
        else:
            setattr(cls, f.name, f.default)
    pos_fields = {} # type: Dict[str, Field]
    kw_fields = {} # type: Dict[str, Field]
    for n, f in fields.items(): 
        if f.kw_only: kw_fields[n] = f
        else: pos_fields[n] = f
    return fields, pos_fields, kw_fields

def _make_field_init(clazz, self_name, field):
    lines = []
    make_setter = lambda value: f"object.__setattr__({self_name},\"{field.name}\",{value})"
    if field.required: 
        lines.append(make_setter(field.name))
    else:
        lines.append(f"if {field.name} is not MISSING: " + make_setter(field.name))
    if field.default_factory is not MISSING:
        lines.append(f"else: " + make_setter(f"{self_name}.__struct_fields__['{field.name}'].default_factory()"))
    elif field.initializer is not MISSING:
        lines.append(f"else: " + make_setter(f"{self_name}.__struct_fields__['{field.name}'].initializer({self_name})"))
    return lines

def _make_init(clazz, fields, pos_fields, kw_fields, globals):
    self_name = "__struct_self__" if "self" in fields else "self"
    params = clazz.__struct_params__
    args = []
    body_lines = []
    for f in pos_fields.values():
        if f.required:
            args.append(f"{f.name}")
        else:
            args.append(f"{f.name}=MISSING")
    if kw_fields:
        args.append("*")
        for f in kw_fields.values():
            args.append(f"{f.name}=MISSING")
    for f in fields.values():
        body_lines.extend(_make_field_init(clazz, self_name, f))
    return _create_fn(
        "__init__",
        [self_name] + args,
        body_lines,
        locals={"cls": clazz, "MISSING": MISSING},
        globals=globals
    )

def _make_frozen_setattr(cls):
    from stanza.util import FrozenInstanceError
    return _create_fn(
        "__setattr__",
        ["self", "name", "value"],
        ["raise FrozenInstanceError(f'cannot set \"{name}\" on a frozen {cls.__name__}')"],
        locals={"FrozenInstanceError": FrozenInstanceError, "cls": cls}
                
    )

# UTILITIES


# ported from original dataclasses.py file, utility to create
# a function with a given body and signature
def _create_fn(name, args, body, *, globals=None, locals=None,
               return_type=MISSING):
    if locals is None:
        locals = {}
    return_annotation = ''
    if return_type is not MISSING:
        locals['__struct_return_type__'] = return_type
        return_annotation = '->__struct_return_type__'
    args = ','.join(args)
    if not body:
        body = ["pass"]
    body = '\n'.join(f'  {b}' for b in body)
    txt = f' def {name}({args}){return_annotation}:\n{body}'
    local_vars = ', '.join(locals.keys())
    txt = f"def __create_fn__({local_vars}):\n{txt}\n return {name}"
    ns = {}
    exec(txt, globals, ns)
    return ns['__create_fn__'](**locals)