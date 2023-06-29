"""
    Contains high-level "target" entrypoint
"""
import os
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=10'

import sys
import asyncio
from stanza.dataclasses import dataclass, field
from stanza.runtime.database import Database
from stanza.runtime.container import Target
from stanza.runtime.pool import Pool
from stanza.dataclasses.arg import \
    ArgParser, ArgParseError, flag

from typing import List

@dataclass
class EmptyConfig:
    pass

@dataclass
class ActivityConfig:
    entrypoint: str = field(arg_positional=True)
    help: bool = field(nargs=0, default=False,
            arg_help="Prints this help message",
            arg_builder=flag())
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

    parser = ArgParser(ActivityConfig)
    activity_info, = parser.parse(sys.argv[1:], ignore_unknown=True)
    entrypoint = activity_info.entrypoint
    entrypoint = _load_entrypoint(entrypoint)
    if not hasattr(entrypoint, "__config__"):
        raise ArgParseError("Entrypoint must be decorated with @activity")
    activity_conf_cls = entrypoint.__config__
    parser.add_to_parser(activity_conf_cls)
    _,  config = parser.parse(sys.argv[1:])

    if activity_info.help:
        parser.print_help()
        return
    entrypoint(config, None)

# Entrypoint for launching activities, sweeps
def launch_activity():
    try:
        asyncio.run(launch_activity_main())
    except ArgParseError as e:
        print(e)
    except KeyboardInterrupt:
        sys.exit(0)