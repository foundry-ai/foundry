import tempfile
import rpyc
import subprocess
import threading
import cloudpickle
import os
import uuid
import queue
import itertools

from stanza.logging import logger
from dataclasses import dataclass

from rpyc.utils.server import ThreadedServer
from rpyc.utils.helpers import classpartial, async_

class Context:
    def __init__(self, fs, walker=None):
        self.fs = fs
        self.walker = walker

    # In charge of launching a process within
    # this context
    def spawn(self, args):
        if not isinstance(self.fs, OSFS):
            raise RuntimeError(
                "To spawn process, filesystem must be of type OSFS"
            )
        l = ["poetry", "run"]
        l.extend(args)
        return subprocess.Popen(l, cwd=self.fs.root_path)
    
    def mirror_to(self, other):
        logger.info("Mirroring filesystem")
        fs.mirror.mirror(self.fs, other.fs, walker=self.walker)
        install_proc = other.spawn(["poetry", "install"])
        install_proc.wait()
        logger.info("Done mirroring")

    # By default close() does nothing
    def close(self):
        self.fs.close()

    # Will find the root poetry directory
    # and use the standard excludes
    @staticmethod
    def default():
        return Context(OSFS("."), walker=Walker(
            exclude=["poetry.toml"],
            exclude_dirs=[".git", ".venv", "*__pycache__"])
        )
    
class TempContext(Context):
    # Create an empty temp context
    def __init__(self):
        directory = tempfile.mkdtemp(prefix="stanza_")
        super().__init__(OSFS(directory))
    
    def close(self):
        # Delete temporary directory
        pass

# Represents the interface to a worker.
# Basically you can send it pickled functions
# and it will give you a reference to execute that function
class Worker:
    def create_executor(self, pickled_fun):
        # unpickle the function on this side of the network
        fun = cloudpickle.loads(pickled_fun)
        return fun

LOCAL_MANAGER = None
def local_spawner():
    global LOCAL_MANAGER
    if LOCAL_MANAGER is None:
        LOCAL_MANAGER = WorkerManager(18860)
        LOCAL_MANAGER.start(daemon=True)
    
    conn = rpyc.connect("localhost", 18860,
        config={'allow_public_attrs': True})
    return conn.root
    # return WorkerSpawnService(LOCAL_MANAGER)

# All spawner services
# share a worker manager
class WorkerManager:
    def __init__(self, service_port):
        self.service_port = service_port
        # id --> 
        self.workers = {}
        self.startup_callbacks = {}
    
    def register_worker(self, id, worker):
        logger.info(f"Registering worker [yellow]{id}[/yellow]")
        self.workers[id] = worker
        cbs = self.startup_callbacks.get(id, [])
        del self.startup_callbacks[id]
        for c in cbs:
            c(worker)
    
    def add_startup_callback(self, id, callback):
        self.startup_callbacks.setdefault(id, []).append(callback)

    def spawn_workers(self, context, n):
        # Copy the context
        logger.trace(f"Spawning {n} workers")
        ports = []

        workers = []
        done = threading.Event()

        def cb(worker):
            workers.append(worker)
            if len(workers) == n:
                done.set()
        for i in range(n):
            id = str(uuid.uuid4())
            self.add_startup_callback(id, cb)
            proc = context.spawn(
                ["python", "-m", "stanza.experiment.launch_worker",
                "--id", id, "--manager_port", f"{self.service_port}"]
            )
        done.wait()
        logger.trace("All workers initialized")
        return workers
    
    def spawner(self):
        return WorkerSpawnService(self)
    
    def start(self, daemon=False):
        server = ThreadedServer(
            classpartial(WorkerSpawnService, self),
            port=self.service_port,
            protocol_config={
                'allow_public_attrs': True
            }
        )
        thread = threading.Thread(target=server.start, daemon=daemon)
        thread.start()
        return self

class WorkerSpawnService(rpyc.Service):
    def __init__(self, manager):
        self._manager = manager
        os.makedirs("/tmp/stanza", exist_ok=True)
        self.context = Context(OSFS("/tmp/stanza"))
        # Keep track of the workers
        # we have spawned, and terminate them
        # upon disconnect of the person using the
        # spawner service
        self.workers = []

    def __del__(self):
        # If the remote process disconnects,
        # terminate all workers associated
        # with this service
        pass
    
    # NOTE: register_worker is only called by workers
    # and is therefore on a different instance
    def register_worker(self, id, worker):
        self._manager.register_worker(id, worker)

    def spawn_workers(self, n):
        workers = self._manager.spawn_workers(self.context, n)
        self.workers.extend(workers)
        return workers

# Client-side utils

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
                self.conns.append((local_spawner(), c.workers))
            else:
                conn = rpyc.connect(c.hostname, c.port,
                    config={'allow_public_attrs': True})
                self.conns.append((conn.root, c.workers))

        # spawn all of the workers
        self.workers = []
        for (spawner, num_workers) in self.conns:
            self.context.mirror_to(spawner.context)
            self.workers.extend(spawner.spawn_workers(num_workers))
        
    def run(self, func, iterable):
        executors = queue.Queue()
        for w in self.workers:
            executors.put(w.create_executor(cloudpickle.dumps(func)))
        for v in iterable:
            e = executors.get()
            res = async_(e)(v)
            def cb(res):
                executors.put(e)
            res.add_callback(cb)

    def map(self, func, iterable):
        # shove the specified function
        # over the network to all of the workers
        # one time, they return a netref of the
        # deserialized function from the other side
        executors = queue.Queue()
        for w in self.workers:
            executors.put(w.create_executor(cloudpickle.dumps(func)))

        results = []
        pending = []
        for i, v in enumerate(iterable):
            e = executors.get()
            res = async_(e)(v)
            def cb(res):
                results.append((i, res.value))
                # free up the executor
                executors.put(e)
            res.add_callback(cb)
            pending.append(res)
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

import datetime
# Fix the comparison for the mirror operation
def _copy_datetime(date):
    return datetime.datetime(year=date.year, month=date.month,
                day=date.day, hour=date.hour, minute=date.minute,
                second=date.second, microsecond=date.microsecond)

def _patched_compare(info1, info2):
    # type: (Info, Info) -> bool
    """Compare two `Info` objects to see if they should be copied.

    Returns:
        bool: `True` if the `Info` are different in size or mtime.

    """
    # Check filesize has changed
    if info1.size != info2.size:
        return True
    # Check modified dates
    date1 = info1.modified
    date2 = info2.modified
    date1 = _copy_datetime(date1)
    date2 = _copy_datetime(date2)
    return date1 is None or date2 is None or date1 > date2

import fs.mirror as mirror
mirror._compare = _patched_compare