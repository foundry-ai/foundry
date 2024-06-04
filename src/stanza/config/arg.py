import argparse
from . import ConfigProvider, NO_DEFAULT

from typing import Literal, Sequence
from rich.text import Text

class Arguments:
    def __init__(self, args):
        self.args = args
        self.options = []

    def add_option(self, name : str, desc: str):
        self.options.append((name, desc))
    
    def parse_option(self, name : str, desc: str):
        self.add_option(name, desc)
        parser = argparse.ArgumentParser(add_help=False, prog="")
        parser.add_argument(f"--{name}", required=False, type=str)
        ns, args = parser.parse_known_intermixed_args(self.args)
        self.args = args
        val = getattr(ns, name)
        return val

class ArgConfig(ConfigProvider):
    def __init__(self, args : Arguments, prefix=None, active=True, cases=None):
        self._args = args
        self._prefix = prefix
        self._cases = cases or []
        self._active = active

    def get(self, name: str, type: str, desc: str, default=NO_DEFAULT):
        if self._prefix:
            name = f"{self._prefix}_{name}"

        if not self._active:
            self._args.add_option(name, desc)
            if default is NO_DEFAULT:
                return type()
            return default
        else:
            arg = self._args.parse_option(name, desc)
            if arg is None and default is NO_DEFAULT:
                raise ValueError(f"Argument {name} not provided")
            if arg is None:
                return default
            if type is bool:
                arg = arg.lower()
                return arg == "true" or arg == "t" or arg == "y"
            return type(arg)
    
    def scope(self, name: str, desc: str) -> "ConfigProvider":
        prefix = name if not self._prefix else f"{self._prefix}_{name}"
        return ArgConfig(self._args, prefix)
    
    def case(self, name: str, desc: str, active: bool):
        return ArgConfig(self._args, self._prefix, active, self._cases + [name])