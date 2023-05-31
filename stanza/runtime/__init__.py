"""
    Contains high-level "target" entrypoint
"""
import os
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=10'

import functools
import sys
import asyncio
import rich
from stanza.util.logging import logger
from stanza.util.dataclasses import dataclass
from stanza.runtime.config import RuntimeParser, ArgParseError
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
    
    def run(self, config=None, database=None, *args, **kwargs):
        if database is None:
            from stanza.runtime.database.dummy import DummyDatabase
            database = DummyDatabase()
        if config is None:
            config = self.config_dataclass()
        return self._exec(config, database, *args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)

# A decorator version for convenience
def activity(config_dataclass, f=None):
    if callable(f):
        return FuncActivity(config_dataclass, f)

    # return the decorator
    def decorator(f):
        a = FuncActivity(config_dataclass, f)
        return functools.wraps(f)(a)
    return decorator

def activity_sub(entrypoint, db_url, cfg):
    db = Database.from_url(db_url)
    entrypoint(cfg, db)

async def launch_activity_main():
    import sys
    import os
    # Add projects to python path
    sys.path.append(os.path.abspath(os.path.join(__file__, "..","..","..", "projects")))

    parser = RuntimeParser()
    try:
        runtime_cfg = parser.parse_args(sys.argv[1:])
    except ArgParseError as e:
        rich.print(f"[red]Invalid Arguments:[/red] {e}")
        return
    target = await Target.from_url(runtime_cfg.target)
    async with Pool(target) as p:
        activity = functools.partial(activity_sub, runtime_cfg.activity, runtime_cfg.database)
        await p.run(activity, runtime_cfg.configs)

# Entrypoint for launching activities, sweeps
def launch_activity():
    try:
        asyncio.run(launch_activity_main())
    except KeyboardInterrupt:
        sys.exit(0)