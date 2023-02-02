import numpy as np
from stanza.logging import logger
from dataclasses import dataclass

import jax.numpy as jnp

# TODO: No idea what this API should look like yet

# Type hint annotations
class Figure:
    def __init__(self, fig):
        self.fig = fig

class Video:
    def __init__(self, data, fps=4):
        self.data = data
        self.fps = fps

class Repo:
    def experiment(self, name):
        pass

    @staticmethod
    def from_url(repo_url):
        if repo_url == 'dummy':
            from stanza.experiment.dummy import DummyRepo
            return DummyRepo()
        elif repo_url.startswith('wandb/'):
            entity = repo_url[6:]
            from stanza.experiment.wandb import WandbRepo
            return WandbRepo(entity)

class Experiment:
    def create_run(self, name=None):
        pass

def remap(obj, type_mapping):
    if isinstance(obj, dict):
        return { k: remap(v, type_mapping) for (k,v) in obj.items() }
    elif isinstance(obj, list):
        return [ remap(v, type_mapping) for v in obj ]
    elif isinstance(obj, tuple):
        return tuple([ remap(v, type_mapping) for v in obj ])
    elif isinstance(obj, jnp.ndarray):
        return np.array(obj)
    elif type(obj) in type_mapping:
        return type_mapping[type(obj)](obj)
    else:
        return obj

# Helper function to merge 
def _merge(a, b, path=None):
    if path is None: path = []
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                merge(a[key], b[key], path + [str(key)])
            else:
                a[key] = b[key]
        else:
            a[key] = b[key]
    return a

class Run:
    def __init__(self):
        self.step = 0
        self._temp_data = {}

    def log(self, data, step=None, commit=True):
        if step is None and commit:
            self._log(data)
        elif step == self.step or not commit:
            merge(self._temp_data, data)
        elif step == self.step + 1:
            self._log(self._temp_data)
            self._temp_data = {}
            merge(data, self._temp_data)
        else:
            raise RuntimeError("Step must be either current or next!")

    def _log(self, data):
        raise RuntimeError("Not implemented!")


# Distribute run tools
from .host import Context, WorkerManager
import cloudpickle
import itertools
import queue
import rpyc

@dataclass
class PoolInfo:
    workers: int
    hostname: str = None # if none, a worker_spawner will be launched
    port: int = 18860

# Specify the connections to connect to 
# and the number of workers to spin up per connection
class WorkerPool:
    def __init__(self, context, *conn_infos):
        self.context = context
        # connect to each machine specified
        # if the hostname is None, use a local spawner
        self.conns = []
        self.local_server = None
        for c in conn_infos:
            if c.hostname is None:
                self.local_server = self.local_server or \
                    WorkerManager(18860).start()

                self.conns.append((self.local_server, c.workers))
            else:
                conn = rpyc.connect(c.hostname, c.port)
                self.conns.append((conn.root, c.workers))

        # spawn all of the workers
        self.workers = []
        for (spawner, workers) in self.conns:
            workers.extend(spawner.spawn_workers(self.context, workers))

    def map(func, iterable):
        # shove the specified function
        # over the network to all of the workers
        # one time, they return a netref of the
        # deserialized function from the other side
        executors = queue.Queue()
        for w in self.workers:
            executors.put(w.create_executor(cloudpickle.dumps(func)))

        results = []
        pending = []
        for i in itertools.count():
            e = executors.poll()
            try:
                v = next(iterable)
            except StopIteration:
                break
            res = async_(e)(v)
            def cb(res):
                results.append((i, res))
                executors.put(e)
            res.add_callback(cb)
            pending.add(res)
        # Wait for all results to finish
        for p in pending:
            p.wait()
        # sort by i
        results.sort(key=lambda x: x[0])
        # get rid of the index
        results = [v for (_, v) in results]
        # return the results
        return results

    def close(self):
        pass

    def __enter__(self):
        return self
    
    def __exit__(self, type, value, traceback):
        self.close()