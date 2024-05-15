from typing import Callable, Type, Generic, TypeVar, Any
from rich.console import RenderableType
from rich.text import Text
from rich.table import Table

from .util import get_struct_docstrings

import typing
import argparse
import sys
import rich

from .parser import OptionParser

from stanza.struct.format import Context, Format, FormatProvider
from stanza import struct

class ArgParseError(Exception):
    pass

class Command:
    def __init__(self, callback: Callable,
                       arg_type: Type,
                       command_name: str = None,
                       add_help: bool = True,
                       format_name: str = "cli",
                       format_providers: dict[str, FormatProvider] = {}):
        self.callback = callback
        self.command_name = command_name
        self.arg_type = arg_type
        self.add_help = add_help

        self.format_name = format_name
        self.format_providers = dict(format_providers)
        if not self.format_name in self.format_providers:
            self.format_providers[self.format_name] = DefaultFormatProvider()

    def __call__(self, cli_args: list[str] = None):
        if cli_args is None:
            import sys
            cli_args = sys.argv
        cli_args = list(cli_args)
        command_name = self.command_name or (cli_args[0] if cli_args else None)
        ctx = Context(self.format_name, ArgConfig(
            add_help=self.add_help
        ), self.format_providers)
        format = ctx.format_for(self.arg_type)
        parser = format.parser()
        res = parser(cli_args)
        if cli_args:
            left = " ".join(cli_args)
            rich.print(f"[red]Unknown arguments: [/red]{left}")
            rich.print(format.help(command_name))
            return
        return self.callback(res)

def command(arg_type, name=None, add_help=True, format_name="args", format_providers={}):
    def decorator(func):
        return Command(func, arg_type, name, add_help, format_name, format_providers)
    return decorator

import dataclasses

@dataclasses.dataclass
class ArgConfig:
    add_help: bool = True

    prefix: str = ""
    name: str = None
    help_str: str = None
    default: typing.Any = struct.UNDEFINED
    required: bool | None = None

T = TypeVar('T')

class Parser(Generic[T]):
    def __call__(self, args: list[str]) -> T:
        ...

class CliFormat(Format[T, list[str], None], Generic[T]):
    def __init__(self, ctx: Context):
        config = ctx.config
        self.prefix = config.prefix
        self.name = config.name
        self.full_name = self.prefix + self.name if self.name else None
        self.default = config.default
        self.required = config.required if config.required is not None else \
                        config.default is struct.UNDEFINED
        self.help_str = config.help_str

        self._command_add_help = ctx.config.add_help

    def add_to_parser(self, parser: OptionParser):
        raise NotImplementedError

    def handle_results(self, ns: argparse.Namespace, default: Any = struct.UNDEFINED) -> T:
        raise NotImplementedError

    def help_entries(self) -> list[tuple[str, RenderableType, RenderableType]]:
        return [(f"--{self.full_name}", self.help_str, None)]
    
    def help(self, command_name) -> RenderableType:
        parser = OptionParser()
        self.add_to_parser(parser)
        text = Text()
        text.append("Usage: ", style="bold")
        text.append_text(parser.short_usage(command_name))
        if self.help_str:
            text.append("\n\n")
            text.append(self.help_str.trim())
        return text

    def parse(self, args: list[str]) -> T:
        # must at least specify the program name!
        assert len(args) > 0
        parser = OptionParser()
        self.add_to_parser(parser)
        if self._command_add_help:
            parser.add_option("help", nargs=0)
        res, rem = parser.parse(args[1:])
        if self._command_add_help and res.help:
            rich.print(self.help(args[0]))
            sys.exit(0)
        res = self.handle_results(res, None)
        # replace the argument
        # list with the remaining args
        args.clear()
        args.extend(rem)
        return res

    def dump(self, ctx: Context, value: T) -> None:
        raise NotImplementedError

    def parser(self) -> Callable[[list[str]], T]:
        return self.parse

class StructFormat(CliFormat[T], Generic[T]):
    def __init__(self, ctx: Context, cls: Type[T],
                 inline=False, struct_option=True,
                 field_format_transforms={}):
        super().__init__(ctx)
        self.cls = cls
        self.inline = inline
        self.struct_option = struct_option
        self.field_formats : dict[str, CliFormat] = {}
        prefix = (
            self.prefix
            if inline or not self.name else 
            self.prefix + self.name + "."
        )
        for f in struct.fields(cls):
            # if we have a default value for the whole struct
            # from the parent context, use that
            if self.default is not struct.UNDEFINED:
                default = getattr(self.default, f.name)
            elif not self.required: 
                # if this struct is not required in the parent ctx,
                # the true default may not align with f.default
                default = struct.UNDEFINED
            else:
                # note: if the default is based on initializer
                # we can't instantiate it
                if f.default_factory is not struct.UNDEFINED:
                    default = f.default_factory()
                else:
                    default = f.default
            f_type = type(default) if default is not struct.UNDEFINED and default is not None else f.type

            c = ctx.with_config(dataclasses.replace(ctx.config,
                prefix=prefix,
                default=default,
                required=f.required and default is struct.UNDEFINED,
                name=f.name
            ))
            format = c.format_for(f_type)
            if f.name in field_format_transforms:
                format = field_format_transforms[f.name](format)
            self.field_formats[f.name] = format
    
    def add_to_parser(self, parser: OptionParser):
        if self.struct_option and self.full_name:
            parser.add_option(self.full_name, nargs=1)
        for f in self.field_formats.values():
            f.add_to_parser(parser)
    
    def handle_results(self, ns: argparse.Namespace, default: Any = struct.UNDEFINED) -> T:
        default = self.default if default is struct.UNDEFINED else default
        if default is not struct.UNDEFINED and default is not None:
            args = {}
            for (n, f) in self.field_formats.items():
                dv = getattr(default, n, struct.UNDEFINED)
                v = f.handle_results(ns, dv)
                if v is not struct.UNDEFINED:
                    args[n] = v
            return struct.replace(default, **args)
        else:
            # TODO: handle default being None
            # build a new instance
            instance = self.cls.__new__(self.cls)
            for field in struct.fields(self.cls):
                format = self.field_formats[field.name]
                if field.default is not struct.UNDEFINED: v = field.default
                elif field.default_factory is not struct.UNDEFINED: v = field.default_factory()
                elif field.initializer is not struct.UNDEFINED: v = field.initializer(instance)
                else: v = struct.UNDEFINED
                if format is not None:
                    v = format.handle_results(ns, v)
                if v is struct.UNDEFINED:
                    raise ArgParseError(f"Missing required field {field.name}")
                object.__setattr__(instance, field.name, v)
            return instance
    
    def help_entries(self) -> list[tuple[str, RenderableType, RenderableType]]:
        entries = []
        for f in self.field_formats.values():
            entries.extend(f.help_entries())
        return entries

    def help(self, command_name : str) -> RenderableType:
        desc = super().help(command_name)
        table = Table(box=None)
        table.add_column("Options", justify="left")
        table.add_column(justify="left")
        entries = self.help_entries()
        for name, help, desc in entries:
            if name is not None or help is not None:
                table.add_row(name, help)
            if desc is not None:
                table.add_row("", desc)
        return table

class OptionFormat(CliFormat[T], Generic[T]):
    def __init__(self, ctx : Context, type: Type[T], nargs=1):
        super().__init__(ctx)
        self.type = type
        self.base_type = type
        if type == typing.Optional[type]:
            self.base_type = typing.get_args(type)[0]
        self.nargs = nargs
    
    def add_to_parser(self, parser: OptionParser):
        parser.add_option(name=self.full_name, nargs=self.nargs)
    
    def handle_results(self, ns: argparse.Namespace, default=struct.UNDEFINED) -> T:
        default = default if default is not struct.UNDEFINED else self.default
        v = getattr(ns, self.full_name)
        if v is None:
            if self.required:
                raise ArgParseError(f"Missing required option {self.full_name}")
            else:
                return default
        else:
            if len(v) > 1:
                raise ArgParseError(f"Too many values for option {self.full_name}")
            if self.base_type == bool:
                v = v[0].lower()
                return v == "t" or v == "true" or v == "y" or v == "yes"
            return self.base_type(v[0])
    
class DefaultFormatProvider:
    def __call__(self, ctx: Context, type: Type[T]) -> CliFormat[T]:
        if struct.is_struct(type):
            return StructFormat(ctx, type)
        else:
            return OptionFormat(ctx, type)