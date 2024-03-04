import argparse
from typing import Literal, Sequence
from rich.text import Text

class OptionParser:
    def __init__(self):
        self._parser = argparse.ArgumentParser(add_help=False, prog="")
        self._has_options = False
        self._arguments = []

    def short_usage(self, name="") -> Text:
        text = Text()
        if name:
            text.append(name)
            text.append(" ")
        for arg in self._arguments:
            text.append(arg, style="italic")
            text.append(" ")
        if self._has_options:
            text.append("[OPTIONS]", style="italic")
        return text
    
    def add_argument(self, name : str, nargs : int | Literal["*"] = 0):
        self._parser.add_argument(name,
            required=False, nargs=nargs
        )
        self._arguments.append(name)

    def add_option(self, name : str = "", nargs : int | Literal["*"] = 0):
        if nargs > 0 or nargs == "*":
            self._parser.add_argument(f"--{name}", required=False, type=str, nargs=nargs)
        else:
            self._parser.add_argument(f"--{name}", required=False, action="store_true", default=False)
        self._has_options = True

    def parse(self, args : Sequence[str]) -> tuple[argparse.Namespace, list[str]]:
        ns, remaining = self._parser.parse_known_args(args)
        return ns, remaining