import os
import sys
import rich
import rich.jupyter
import rich.logging
import rich.terminal_theme
import rich._log_render
import multiprocessing
from rich.logging import RichHandler

from pathlib import Path


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
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ["WANDB_SILENT"] = "true"
    if entrypoint is None:
        if len(sys.argv) < 2:
            print("[red]Must specify entrypoint[/red]")
            sys.exit(1)
        entrypoint_str = sys.argv[1]
        entrypoint = _load_entrypoint(entrypoint_str)
    setup()
    logger.info(f"Launching {entrypoint_str}")
    # remove the "launch" argument
    sys.argv.pop(0)
    entrypoint()