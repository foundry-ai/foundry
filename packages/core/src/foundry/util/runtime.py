import os
import sys
import rich
import rich.jupyter
import rich.logging
import rich.terminal_theme
import rich._log_render
import subprocess
import multiprocessing
import argparse
import abc
import functools
import typing

from typing import Literal, Sequence, Callable, Type
from rich.text import Text
from rich.logging import RichHandler
from pathlib import Path
from bdb import BdbQuit

import logging
logger = logging.getLogger("foundry")

# Make not expand
class CustomLogRender(rich._log_render.LogRender):
    def __call__(self, *args, **kwargs):
        output = super().__call__(*args, **kwargs)
        if not self.show_path:
            output.expand = False
        return output

LOGGING_SETUP = False
def setup_logger(show_path=True):
    global LOGGING_SETUP
    FORMAT = "%(name)s - %(message)s"
    if not LOGGING_SETUP:
        console = rich.get_console()
        handler = RichHandler(
            markup=True,
            rich_tracebacks=True,
            show_path=show_path,
            console=console
        )
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
    logger = logging.getLogger("foundry")
    logger.setLevel(logging.DEBUG)
    # Prevent "retrying" warnings from connectionpool
    # if running wandb offline
    logger = logging.getLogger("urllib3.connectionpool")
    logger.setLevel(logging.ERROR)

SETUP_JAX_CACHE = False

def setup_jax_cache():
    global SETUP_JAX_CACHE
    if SETUP_JAX_CACHE:
        return
    from jax.experimental.compilation_cache import compilation_cache as cc
    import tempfile
    user = os.environ.get("USER", "foundry")
    JAX_CACHE = Path(tempfile.gettempdir()) / f"jax_cache_{user}"
    JAX_CACHE = Path(os.environ.get("JAX_CACHE", JAX_CACHE))
    JAX_CACHE.mkdir(parents=True, exist_ok=True)
    cc.initialize_cache(str(JAX_CACHE))
    SETUP_JAX_CACHE = True

SETUP_GC = False

def setup_gc():
    global SETUP_GC
    if SETUP_GC:
        return
    import gc
    from jax._src.lib import _xla_gc_callback
    # pop the xla gc callback
    if gc.callbacks[-1] is _xla_gc_callback:
        gc.callbacks.pop()
    SETUP_GC = True

def setup():
    # Enable 64 bit dtypes by default,
    # but make the default dtype 32 bits
    os.environ["JAX_ENABLE_X86"] = "True"
    os.environ["JAX_DEFAULT_DTYPE_BITS"] = "32"

    jupyter = rich.get_console().is_jupyter
    os.environ["WANDB_SILENT"] = "true"
    # The driver path, for non-nix python environments
    if jupyter:
        from foundry.util.ipython import setup_rich_notebook_hook
        setup_rich_notebook_hook()
    setup_logger(False)
    setup_jax_cache()
    setup_gc()