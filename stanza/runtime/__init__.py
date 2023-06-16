"""
    Contains high-level "target" entrypoint
"""
import os
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=10'

import sys
import asyncio
from dataclasses import dataclass, field, is_dataclass
from stanza.runtime.database import Database
from stanza.runtime.container import Target
from stanza.runtime.pool import Pool

from typing import List
import argparse

class ArgParseError(Exception):
    pass

def arg(positional=False, nargs=None, **kwargs):
    return field(
        metadata=dict(
            positional=positional,
            nargs=nargs
        ), **kwargs)

class Arg:
    def __init__(self, name, type_, positional=False, nargs=None, 
                 default=None, default_factory=None):
        self.name = name
        self.type = type_
        self.positional = positional
        self.nargs = nargs
        self.default = default
    
    @staticmethod
    def from_field(field):
        return Arg(field.name, field.type, **field.metadata)

def _add_to_argparser(parser, type_,
                      prefix, name,
                      positional=False,
                      nargs=1, action=None,
                      default=None):
    if is_dataclass(type_):
        if positional:
            builders = []
            for f in type_.__dataclass_fields__.values():
                b = _add_to_argparser(parser, f.type,
                    prefix=f"{prefix}{name}." if name or prefix else "",
                    name=f.name, positional=positional, nargs=nargs)
                builders.append(b)
            def build_dc(args):
                return type_(**{
                    f.name: b(args) for f, b in zip(type_.__dataclass_fields__.values(), builders)
                })
            return build_dc
        else:
            raise ArgParseError("Cannot add non-positional dataclass to argparser")
    else:
        if positional:
            parser.add_argument(f"{prefix}{name}",
                                default=default, nargs=nargs)
        else:
            parser.add_argument(f"--{prefix}{name}",
                                default=default, nargs=nargs)
        return lambda a: getattr(a,f"{prefix}{name}")


def make_argparser(cls):
    p = argparse.ArgumentParser(add_help=False)
    build_dc = _add_to_argparser(p, cls, "", "", positional=True)
    def parse(args, return_unknown=False):
        if return_unknown:
            parsed, remainder = p.parse_known_args(args)
            dc = build_dc(parsed)
            return dc, remainder
        else:
            parsed = p.parse_args(args)
            dc = build_dc(parsed)
            return dc
    return parse

@dataclass
class EmptyConfig:
    pass

@dataclass
class ActivityConfig:
    entrypoint: str = arg(positional=True)
    help: bool = arg(nargs=0, default=False)
    database: str = None
    target: str = None
    py_config: str = None
    json_config: str = None

# Labels a function with a config
def activity(config_class, f=None):
    if f is not None:
        f.__config__ = config_class
        return f
    else:
        def dec(f):
            return activity(config_class, f)
        return dec

def _load_entrypoint(entrypoint_string):
    import importlib
    parts = entrypoint_string.split(":")
    if len(parts) != 2:
        raise ArgParseError("Entrypoint must include module and activity")
    module, attr = parts
    module = importlib.import_module(module)
    return getattr(module, attr)

async def launch_activity_main():
    import sys
    import os
    # Add projects to python path
    sys.path.append(os.path.abspath(os.path.join(__file__, "..","..","..", "projects")))

    parser = ActivityConfig.build_argparser()
    activity_info, other_args = parser(sys.argv[1:], return_unknown=True)
    entrypoint = activity_info.entrypoint
    entrypoint = _load_entrypoint(entrypoint)
    if not hasattr(entrypoint, "__config__"):
        raise ArgParseError("Entrypoint must be decorated with @activity")
    activity_conf_cls = entrypoint.__config__
    parser = activity_conf_cls.build_argparser()
    activity_config = parser(other_args)
    print(activity_config)

# Entrypoint for launching activities, sweeps
def launch_activity():
    try:
        asyncio.run(launch_activity_main())
    except ArgParseError as e:
        print(e)
    except KeyboardInterrupt:
        sys.exit(0)