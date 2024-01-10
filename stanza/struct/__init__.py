import sys
import inspect

from typing import Dict, NamedTuple

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

def is_frozen(struct):
    return getattr(struct, "__frozen__", False)

# Returns a frozen version of the struct (a new instance)
def freeze(struct):
    if getattr(struct, "__frozen__", False):
        return struct
    else:
        # copy the fields to a new instance
        # and set the frozen flag
        i = struct.__class__.__new__()
        for n, f in struct.__struct_fields__.items():
            setattr(i, n, getattr(struct, f.name))
        object.__setattr__(i, "__frozen__", True)
        return i

def mutate(_struct, **kwargs):
    for k, v in kwargs.items():
        setattr(_struct, k, v)

class StructParams(NamedTuple):
    kw_only: bool
    frozen: bool

def struct(cls=MISSING, *, frozen=False, kw_only=False):
    params = StructParams(kw_only=kw_only, frozen=frozen)
    builder = lambda cls: make_struct(cls, params)
    if cls is not MISSING:
        return builder(cls)
    else:
        return builder

def make_struct(base_struct, params):
    fields, pos_fields, kw_fields = _collect_fields(base_struct, params)

    if base_struct.__module__ in sys.modules:
        globals = sys.modules[base_struct.__module__].__dict__
    else:
        globals = {}
    # create a new class that extends base_struct and has Struct type mixing
    from stanza.lift.struct import StructModule
    class cls(base_struct, StructModule):
        pass
    cls.__name__ = base_struct.__name__
    cls.__struct_params__ = params
    cls.__struct_fields__ = fields
    cls.__slots__ = tuple(field.name for field in fields.values())
    cls.__init__ = _make_init(cls, fields, pos_fields, kw_fields, globals)
    cls.__setattr__ = _make_setattr(cls)

    if not getattr(cls, '__doc__'):
        # Create a class doc-string, same a dataclasses
        try:
            text_sig = str(inspect.signature(cls)).replace(' -> None', '')
        except (TypeError, ValueError):
            text_sig = ''
        cls.__doc__ = (cls.__name__ + text_sig)
    return cls

def _collect_fields(cls, params) -> Dict[str, Field]:
    fields = {}

    # handle inheritance
    for b in cls.__mro__[-1:0:-1]:
        sub_fields = getattr(b, "__struct_fields__", None)
        if sub_fields is not None:
            sub_params = b.__struct_params__
            for f in sub_fields.values():
                fields[f.name] = f
            if sub_params.frozen and not params.frozen:
                raise TypeError("Cannot inherit non-frozen struct from a frozen struct")
            elif not sub_params.frozen and params.frozen:
                raise TypeError("Cannot inherit frozen struct from a non-frozen struct")
    
    annotations = inspect.get_annotations(cls)
    annotation_fields = {}
    for name, _type in annotations.items():
        default = getattr(cls, name, MISSING)
        annotation_fields[name] = Field(
            name, _type, default, 
            MISSING, MISSING
        )

    for name, value in cls.__dict__.items():
        if isinstance(value, Field) and not name in annotation_fields:
            raise TypeError(f"field {name} has no type annotation")

    # add fields from annotations and delete them from the class if default
    for f in annotation_fields.values():
        fields[f.name] = f
        if isinstance(getattr(cls, f.name, None), Field):
            if f.default is MISSING:
                delattr(cls, f.name)
            else:
                setattr(cls, f.name, f.default)
    pos_fields = fields
    kw_fields = {}
    return fields, pos_fields, kw_fields

def _make_field_init(clazz, self_name, field):
    lines = []
    if field.required: 
        lines.append(f"{self_name}.{field.name} = {field.name}")
    else:
        lines.append(f"if {field.name} is not MISSING: {self_name}.{field.name} = {field.name}")
    if field.default_factory is not MISSING:
        lines.append(f"else: {self_name}.{field.name} = {self_name}.__struct_fields__['{field.name}'].default_factory()")
    elif field.initializer is not MISSING:
        lines.append(f"else: {self_name}.{field.name} = {self_name}.__struct_fields__['{field.name}'].initializer({self_name})")
    return lines

def _make_init(clazz, fields, pos_fields, kw_fields, globals):
    self_name = "__struct_self__" if "self" in fields else "self"
    frozen = clazz.__struct_params__.frozen
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
    
    # mark as not frozen by default
    body_lines.append(f"object.__setattr__({self_name},'__frozen__', False)")
    for f in fields.values():
        body_lines.extend(_make_field_init(clazz, self_name, f))
    # if frozen by default
    if frozen:
        body_lines.append(f"object.__setattr__({self_name},'__frozen__', True)")

    # call the StructModule init hook for lifting purposes
    # if name is 
    if "name" in fields:
        body_lines.append("StructModule.__init__(self)")
    else:
        if not kw_fields:
            args.append("*")
        args.append("name=None")
        body_lines.append(f"StructModule.__init__(self, name=name)")

    # print("\n".join(body_lines))
    from stanza.lift.struct import StructModule
    return _create_fn(
        "__init__",
        [self_name] + args,
        body_lines,
        locals={"cls": clazz, "MISSING": MISSING, "StructModule": StructModule},
        globals=globals
    )

def _make_setattr(cls):
    from stanza.struct.frozen import FrozenInstanceError
    from stanza.lift.struct import StructModule
    return _create_fn(
        "__setattr__",
        ["self", "name", "value"],
        ["if getattr(self,'__frozen__', False) and not name.startswith('__mod'): raise FrozenInstanceError(f'cannot set \"{name}\" on a frozen {cls.__name__}')",
        # call the StructModule setattr hook for lifting purposes
        "StructModule.__setattr__(self, name, value)"],
        locals={"FrozenInstanceError": FrozenInstanceError, "cls": cls,
                "StructModule": StructModule}
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