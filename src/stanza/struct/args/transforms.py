from stanza import struct

from .core import StructFormat

def dataclass_params(format="args",
        fallback_default=struct.UNDEFINED,
        required=None,
        **field_format_transforms):
    def decorator(cls):
        format_provider = classmethod(
            lambda cls, ctx: StructFormat(
                cls, ctx,
                required=required,
                default=fallback_default,
                field_format_transforms=field_format_transforms
        ))
        setattr(cls, f"__{format}_format__", format_provider)
        return cls
    return decorator

__all__ = [
    "dataclass_params"
]