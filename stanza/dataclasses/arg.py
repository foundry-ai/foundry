from dataclasses import is_dataclass
import argparse

class ArgParseError(Exception):
    pass

class Arg:
    def __init__(self, name, short, positional, help):
        self.name = name
        self.short = short
        self.positional = positional
        self.help = help

    def add_to_parser(self, parser, prefix=None):
        pass

    def parse_result(self, args, prefix=None):
        pass

class SimpleArg(Arg):
    def __init__(self, parser, name, short=None, positional=False, help=False):
        super().__init__(name, short, positional, help)
        if parser == bool:
            parser = lambda x: (
                x.lower() == "true" or
                x.lower() == "t" or
                x.lower() == "yes" or
                x.lower() == "y"
            )
        self.parser = parser
    
    def add_to_parser(self, parser, prefix=""):
        arg = f"{prefix}{self.name}"
        if not self.positional:
            arg = f"--{arg}"
        parser.add_argument(
            arg, type=str, help=self.help if self.help else ""
        )

    def parse_result(self, args, prefix=None):
        arg = f"{prefix}{self.name}" if prefix else self.name
        val = getattr(args, arg)
        return self.parser(val) if val is not None else None

class FlagArg(Arg):
    def __init__(self, name, short, help=None):
        super().__init__(name, short, False, help)
        self.help = help

    def add_to_parser(self, parser, prefix=""):
        parser.add_argument(
            f"--{prefix}{self.name}", 
            action="store_true", help=self.help if self.help else ""
        )

    def parse_result(self, args, prefix=""):
        arg = f"{prefix}{self.name}"
        val = getattr(args, arg, None)
        return val

# Builders 

def flag():
    def builder(name, short, positional, help):
        return FlagArg(name, short, help)
    return builder

class DataclassArg(Arg):
    def __init__(self, dclass, name, short=None, help=None):
        super().__init__(name, short, True, help)
        self.dclass = dclass
        self.args = []
        for f in dclass.__dataclass_fields__.values():
            name = f.name
            type = f.type
            positional = f.metadata.get('arg_positional', False)
            short = f.metadata.get('arg_short', None)
            builder = f.metadata.get('arg_builder', None)
            help = f.metadata.get('arg_help', None)
            if builder is not None:
                a = builder(name, short, positional, help)
            elif is_dataclass(type):
                a = DataclassArg(type, name, help)
            else:
                a = SimpleArg(type, name, short, positional, help)
            self.args.append(a)

    def add_to_parser(self, parser, prefix=""):
        if self.name is not None:
            prefix = f"{prefix}{self.name}." if prefix else f"{self.name}."
        for a in self.args:
            a.add_to_parser(parser, prefix)
        
    def parse_result(self, args, prefix=""):
        if self.name is not None:
            prefix = f"{prefix}{self.name}." if prefix else f"{self.name}."
        kwargs = {}
        for a in self.args:
            v = a.parse_result(args, prefix)
            if v is not None:
                kwargs[a.name] = v
        return self.dclass(**kwargs)

class ArgParser:
    def __init__(self, *dcs):
        self._args = []
        self._parser = argparse.ArgumentParser(add_help=False)

        for dc in dcs:
            self.add_to_parser(dc)
        
    def print_help(self):
        self._parser.print_help()
    
    def add_to_parser(self, type, help=None):
        arg = DataclassArg(type, None, help)
        arg.add_to_parser(self._parser)
        self._args.append(arg)
    
    def parse(self, args, ignore_unknown=False):
        if ignore_unknown:
            args, _ = self._parser.parse_known_args(args)
        else:
            args = self._parser.parse_args(args)
        results = []
        for a in self._args:
            results.append(a.parse_result(args))
        return results