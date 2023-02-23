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
from stanza.runtime.config import RuntimeParser
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

async def launch_activity_main():
    import sys
    import os
    # Add projects to python path
    sys.path.append(os.path.abspath(os.path.join(__file__, "..","..","..", "projects")))

    parser = RuntimeParser()
    runtime_cfg = parser.parse_args(sys.argv[1:])
    print(runtime_cfg)
    # async with Pool(target) as p:
    #     await p.run(entrypoint, configs)


# Entrypoint for launching activities, sweeps
def launch_activity():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(launch_activity_main())