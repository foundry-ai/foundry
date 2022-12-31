from typing import Any, Iterable, List, Union, NewType
import dataclasses
from functools import partial

def add_to_parser(dataclass, parser, prefix=''):
    for field in dataclasses.fields(dataclass):
        if dataclasses.is_dataclass(field.type):
            add_to_parser(field.type, parser, f'{field.name}.')
        else:
            parser.add_argument(f'--{field.name}', type=field.type,
                default=field.default if field.default is not dataclasses.MISSING else
                        field.default_factory() if field.default_factory is not dataclasses.MISSING else None) 

def from_args(dataclass, args, prefix=''):
    params = {}
    for field in dataclasses.fields(dataclass):
        if dataclasses.is_dataclass(field.type):
            params[field.name] = from_args(field.type, args, f'{field.name}.')
        else:
            params[field.name] = getattr(args, field.name)
    return dataclass(**params)

# Will return a version of the type
# that has add_to_parser and from_parsed
def parsable(dataclass):
    dataclass.add_to_parser = partial(add_to_parser, dataclass)
    dataclass.from_args = partial(from_args, dataclass)
    return dataclass