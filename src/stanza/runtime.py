import os
import sys
import rich
import rich.jupyter
import rich.logging
import rich.terminal_theme
import rich._log_render
import subprocess
import multiprocessing
from rich.logging import RichHandler

from pathlib import Path
from bdb import BdbQuit


import logging
logger = logging.getLogger("stanza")

# Make not expand
class CustomLogRender(rich._log_render.LogRender):
    def __call__(self, *args, **kwargs):
        output = super().__call__(*args, **kwargs)
        if not self.show_path:
            output.expand = False
        return output

LOGGING_SETUP = False
def setup_logger(verbose=0):
    global LOGGING_SETUP
    FORMAT = "%(name)s - %(message)s"
    if not LOGGING_SETUP:
        console = rich.get_console()
        handler = RichHandler(markup=True,
                              rich_tracebacks=True,
                              show_path=not console.is_jupyter)
        renderer = CustomLogRender(
            show_time=handler._log_render.show_time,
            show_level=handler._log_render.show_level,
            show_path=handler._log_render.show_path,
            time_format=handler._log_render.time_format,
            omit_repeated_times=handler._log_render.omit_repeated_times,
        )
        handler._log_render = renderer
        logging.basicConfig(
            level=logging.ERROR,
            format=FORMAT,
            datefmt="[%X]",
            handlers=[handler]
        )
        LOGGING_SETUP = True
    logger = logging.getLogger()
    logger.setLevel(logging.WARNING)
    logger = logging.getLogger("stanza")
    logger.setLevel(logging.DEBUG)

    if verbose > 0:
        jax_logger = logging.getLogger("jax")
        jax_logger.setLevel(logging.DEBUG)
    if verbose < 2:
        jax_logger = logging.getLogger("jax._src.cache_key")
        jax_logger.setLevel(logging.ERROR)

def setup_jax_cache():
    from jax.experimental.compilation_cache import compilation_cache as cc
    JAX_CACHE = Path(os.environ.get("JAX_CACHE", "/tmp/jax_cache"))
    JAX_CACHE.mkdir(parents=True, exist_ok=True)
    cc.initialize_cache(JAX_CACHE)

def setup_gc():
    import gc
    from jax._src.lib import _xla_gc_callback
    # pop the xla gc callback
    if gc.callbacks[-1] is _xla_gc_callback:
        gc.callbacks.pop()

def _load_entrypoint(entrypoint_string):
    import importlib
    parts = entrypoint_string.split(":")
    if len(parts) != 2:
        rich.print("[red]Entrypoint must include module and function[/red]")
        sys.exit(1)
    module, attr = parts
    module = importlib.import_module(module)
    return getattr(module, attr)

def setup():
    cpu_cores = multiprocessing.cpu_count()
    os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={cpu_cores}"
    if rich.get_console().is_jupyter:
        from stanza.util.ipython import setup_rich_notebook_hook
        setup_rich_notebook_hook()
    setup_logger()
    setup_jax_cache()
    setup_gc()

def launch(entrypoint=None):
    try:
        # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        # os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
        os.environ["WANDB_SILENT"] = "true"
        setup()
        if entrypoint is None:
            if len(sys.argv) < 2:
                rich.print("[red]Must specify entrypoint[/red]")
                sys.exit(1)
            entrypoint_str = sys.argv[1]
            entrypoint = _load_entrypoint(entrypoint_str)
        logger.info(f"Launching {entrypoint_str}")
        # remove the "launch" argument
        sys.argv.pop(0)
        entrypoint()
    except (KeyboardInterrupt, BdbQuit):
        # Hard-kill wandb process on manual exit
        cmd = "ps aux|grep wandb|grep -v grep | awk '\''{print $2}'\''|xargs kill -9"
        os.system(cmd)
        logger.error("Exited due to Ctrl-C")
import argparse
from . import ConfigProvider, NO_DEFAULT

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
import abc
import functools
import sys

from typing import Callable, Type
from stanza import dataclasses 


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
        for field in dataclasses.fields(default):
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
        return dataclasses.replace(default, **vals)

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