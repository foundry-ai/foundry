import logging
import os
import sys
from rich import print
from pathlib import Path
import logging
logger = logging.getLogger("stanza")
from rich.logging import RichHandler

def setup_logger():
    FORMAT = "%(name)s - %(message)s"
    logging.basicConfig(
        level=logging.ERROR,
        format=FORMAT,
        datefmt="[%X]",
        handlers=[RichHandler(markup=True, rich_tracebacks=True)]
    )
    logger = logging.getLogger()
    logger.setLevel(logging.ERROR)
    logger = logging.getLogger("stanza")
    logger.setLevel(logging.DEBUG)

def _load_entrypoint(entrypoint_string):
    import importlib
    parts = entrypoint_string.split(":")
    if len(parts) != 2:
        print("[red]Entrypoint must include module and function[/red]")
        sys.exit(1)
    module, attr = parts
    module = importlib.import_module(module)
    return getattr(module, attr)

def launch(entrypoint=None):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ["WANDB_SILENT"] = "true"
    if entrypoint is None:
        if len(sys.argv) < 2:
            print("[red]Must specify entrypoint[/red]")
            sys.exit(1)
        entrypoint_str = sys.argv[1]
        entrypoint = _load_entrypoint(entrypoint_str)
    setup_logger()

    from jax.experimental.compilation_cache import compilation_cache as cc
    JAX_CACHE = Path(os.environ.get("JAX_CACHE", "/tmp/jax_cache"))
    JAX_CACHE.mkdir(parents=True, exist_ok=True)
    cc.initialize_cache(JAX_CACHE)

    logger.info(f"Launching {entrypoint_str}")
    # remove the "launch" argument
    sys.argv.pop(0)
    entrypoint()