import abc
import functools
import sys

from typing import Callable, Type
from stanza import struct 


class NoDefault:
    pass

NO_DEFAULT = NoDefault()

class ConfigProvider(abc.ABC):
    def get(self, name: str, type: Type, desc: str, default=NO_DEFAULT): ...
    def scope(self, name: str, desc: str) -> "ConfigProvider": ...

    # A method that returns a ConfigProvider
    # that only get populated. If active is False
    # the returned ConfigProvider can simply return
    # default values
    def case(self, name: str, desc: str, active: bool) -> "ConfigProvider": ...

    def get_struct(self, default, ignore=set(), flatten=set()):
        vals = {}
        for field in struct.fields(default):
            if field.name in ignore:
                continue
            default_val = getattr(default, field.name)
            type = field.type
            if default_val is not None and hasattr(default_val, "parse"):
                # if there is a parse() method, use it to parse the value
                scope = self.scope(field.name, "") if field.name not in flatten else self
                vals[field.name] = default_val.parse(scope)
            elif (type is bool or type is int or type is float or type is str):
                vals[field.name] = self.get(field.name, field.type, "", default_val)
        return struct.replace(default, **vals)

    def get_cases(self, name: str, desc: str, cases: dict, default: str):
        case = self.get(name, str, desc, default)
        vals = {}
        for case, c in cases.items():
            if hasattr(c, "parse"):
                vals[case] = c.parse(self.scope(name, ""))
            else:
                vals[case] = c
        return vals.get(case, None)

def command(fn: Callable):
    from .arg import Arguments, ArgConfig

    @functools.wraps(fn)
    def main():
        args = Arguments(sys.argv[1:])
        config = ArgConfig(args)
        # config_file = args.parse_option("config", "Path to the configuration file")
        # if config_file is not None:
        #     import yaml
        #     with open(config_file, "r") as f:
        #         y = yaml.load(f)
        #         print(y)
        return fn(config)
    return main