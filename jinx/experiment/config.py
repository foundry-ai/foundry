from typing import Any, Iterable, List, Union, NewType
import dataclasses
from functools import partial

def str_to_bool(value):
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')


def add_to_parser(dataclass, parser, prefix=''):
    for field in dataclasses.fields(dataclass):
        if dataclasses.is_dataclass(field.type):
            add_to_parser(field.type, parser, f'{field.name}.')
        else:
            default = field.default if field.default is not dataclasses.MISSING else \
                field.default_factory() if field.default_factory is not dataclasses.MISSING else None
            if field.type == bool:
                parser.add_argument(f'--{prefix}{field.name}', type=str_to_bool,
                    default=default,
                    required=False)
            else:
                parser.add_argument(f'--{prefix}{field.name}', type=field.type,
                    default=default,
                    required=False) 

def from_args(dataclass, args, prefix=''):
    params = {}
    for field in dataclasses.fields(dataclass):
        if dataclasses.is_dataclass(field.type):
            params[field.name] = from_args(field.type, args, f'{field.name}.')
        else:
            params[field.name] = getattr(args, prefix + field.name)
    return dataclass(**params)

# Will return a version of the type
# that has add_to_parser and from_parsed
def parsable(dataclass):
    dataclass.add_to_parser = partial(add_to_parser, dataclass)
    dataclass.from_args = partial(from_args, dataclass)
    return dataclass