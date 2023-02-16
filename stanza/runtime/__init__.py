"""
    Contains high-level "target" entrypoint
"""
import argparse
import functools
import importlib
import rich
import asyncio

from stanza.util.logging import logger
from stanza.util.dataclasses import dataclass
from stanza.runtime.database import Database
from stanza.runtime.container import Target
from stanza.runtime.pool import Pool

@dataclass
class EmptyConfig:
    pass

# An acivity reports its data to an experiment
class Activity:
    def __init__(self, config_dataclass=None):
        self.config_dataclass = config_dataclass
    
    def run(self, config=None, experiment=None):
        pass

class FuncActivity(Activity):
    def __init__(self, config_dataclass, exec):
        super().__init__(config_dataclass)
        self._exec = exec
    
    def run(self, config=None, run=None, *args, **kwargs):
        if run is None:
            from stanza.runtime.database.dummy import DummyRun
            run = DummyRun()
        if config is None:
            config = self.config_dataclass()
        return self._exec(config, run, *args, **kwargs)

    def __call__(self, config=None, run=None, *args, **kwargs):
        return self.run(config, *args, **kwargs)

# A decorator version for convenience
def activity(config_dataclass=None, f=None):
    if callable(f):
        return FuncActivity(config_dataclass, f)

    # return the decorator
    def decorator(f):
        a = FuncActivity(config_dataclass, f)
        return functools.wraps(f)(a)
    return decorator

def load_entrypoint(entrypoint_string):
    parts = entrypoint_string.split(":")
    if len(parts) != 2:
        raise ValueError("Entrypoint must include module and activity")
    module, attr = parts
    module = importlib.import_module(module)
    return getattr(module, attr)

async def launch_activity_main():
    import sys
    import os
    # Add projects to python path
    sys.path.append(os.path.abspath(os.path.join(__file__, "..","..","..", "projects")))

    parser = argparse.ArgumentParser(add_help=False)
    # If None, will run in current process
    parser.add_argument("--target", default="poetry://localhost", required=False)
    # TODO: use database
    parser.add_argument("--database", default="wandb://", required=False)
    parser.add_argument("entrypoint")
    # add the parse arguments from the activity in question
    args, unknown = parser.parse_known_args()
    # Load the entrypoint
    entrypoint = load_entrypoint(args.entrypoint)
    if not isinstance(entrypoint, Activity):
        print("Entrypoint must be an activity!")
        return
    target = await Target.from_url(args.target)
    # TODO: Load configs
    configs = [None]

    async with Pool(target) as p:
        await p.run(entrypoint, configs)


# Entrypoint for launching activities, sweeps
def launch_activity():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(launch_activity_main())