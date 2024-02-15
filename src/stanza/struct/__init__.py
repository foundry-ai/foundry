import sys
import inspect

from typing import Dict, Tuple, NamedTuple
from typing_extensions import (
  dataclass_transform,  # pytype: disable=not-supported-yet
)

class _MISSING_TYPE:
    pass
MISSING = _MISSING_TYPE()

class Field:
    __slots__ = (
        'name',
        'type',
        'default',
        'default_factory',
        # unlike default_factory, initializer
        # is called with a partially-initialized
        # instance and can be used to set fields
        # based on earlier fields. Either default, default_factory, or initializer
        # must be specified
        'initializer'
    )
    def __init__(self, name, type, default, default_factory, initializer):
        self.name = name
        self.type = type
        self.default = default
        self.default_factory = default_factory
        self.initializer = initializer

    # if this is a required field
    @property
    def required(self):
        return self.default is MISSING and self.default_factory is MISSING and self.initializer is MISSING

def field(*, default=MISSING, default_factory=MISSING, initializer=MISSING):
    return Field(None, None, default, default_factory, initializer)
    
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

@dataclass_transform(field_specifiers=(field,))  # type: ignore[literal-required]
def dataclass(cls=None, *, kw_only=False):
    params = StructParams(kw_only=kw_only)
    builder = lambda cls: make_dataclass(cls, params)
    if cls is not None:
        return builder(cls)
    else:
        return builder

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
    import jax
    from jax.tree_util import GetAttrKey
    def flatten(v):
        children = tuple(getattr(v, f.name) for f in fields(cls))
        return children, None
    def flatten_with_keys(v):
        children = tuple((GetAttrKey(f.name),getattr(v, f.name)) for f in fields(cls))
        return children, None
    def unflatten(_, children):
        i = cls.__new__(cls)
        for f, c in zip(fields(cls), children):
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
            sub_params = b.__struct_params__
            for f in sub_fields.values():
                fields[f.name] = f
    
    annotations = inspect.get_annotations(cls)
    annotation_fields = {}
    for name, _type in annotations.items():
        f = getattr(cls, name, MISSING)
        if not isinstance(f, Field):
            f = Field(None, None, f, MISSING, MISSING)
        # re-instantiate the field with the name and type
        f = Field(name, _type, f.default, f.default_factory, f.initializer)
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
    pos_fields = fields # type: Dict[str, Field]
    kw_fields = {} # type: Dict[str, Field]
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
    if params.kw_only:
        args.append("*")
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