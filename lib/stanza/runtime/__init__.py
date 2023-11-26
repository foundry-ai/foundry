"""
    Contains high-level "target" entrypoint
"""
import multiprocessing
cpus = multiprocessing.cpu_count()
import os
os.environ["XLA_FLAGS"] = f'--xla_force_host_platform_device_count={cpus}'

import sys
import asyncio
from stanza.dataclasses import dataclass, field, is_dataclass
from stanza.reporting import Repo
from stanza.dataclasses.arg import \
    ArgParser, ArgParseError, flag

from typing import List

@dataclass
class EmptyConfig:
    pass

@dataclass
class ActivityConfig:
    entrypoint: str = field(arg_positional=True, default=None)
    help: bool = field(nargs=0, default=False,
            arg_help="Prints this help message",
            arg_builder=flag())
    repo: str = "dummy://"
    target: str = None
    py_config: str = None
    json_config: str = None

# Labels a function with a config
def activity(config_class, f=None):
    if not is_dataclass(config_class):
        raise ValueError("Must specify dataclass")
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

def launch_activity():
    parser = ArgParser(ActivityConfig)
    activity_info, = parser.parse(sys.argv[1:], ignore_unknown=True)
    if activity_info.entrypoint is None:
        parser.print_help()
        return
    entrypoint = activity_info.entrypoint
    entrypoint = _load_entrypoint(entrypoint)
    if not hasattr(entrypoint, "__config__"):
        raise ArgParseError("Entrypoint must be decorated with @activity")
    activity_conf_cls = entrypoint.__config__

    if activity_info.py_config is not None:
        config_entrypoint = _load_entrypoint(activity_info.py_config)
        config = config_entrypoint()
        if not isinstance(config, activity_conf_cls):
            raise ValueError(f"Config must be instance of {activity_conf_cls}")
        # parse any additional args
        parser.add_to_parser(config)
        _,  config = parser.parse(sys.argv[1:])
    else:
        parser.add_to_parser(activity_conf_cls)
        _,  config = parser.parse(sys.argv[1:])

    if activity_info.help:
        parser.print_help()
        return
    repo = Repo(activity_info.repo)
    entrypoint(config, repo)