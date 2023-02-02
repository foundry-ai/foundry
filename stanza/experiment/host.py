import tempfile
import rpyc
import subprocess
import cloudpickle
import os
import uuid

from stanza.logging import logger
from threading import Event

from rpyc.utils.server import ThreadedServer
from rpyc.utils.helpers import classpartial, async_

class Node:
    def __init__(self, parent=None, is_dir=False,
                    children=[], data=None):
        pass

    @staticmethod
    def from_file(path, parent=None):
        pass
    
    @staticmethod
    def from_directory(path, parent=None):
        pass

# Manages a poetry-based project tree in which to launch
# things. Eventually this will be replaced with a Dockerfile + directory
# combination, but poetry + venv seems good enough for now
class Context:
    def __init__(self, root_dir, excludes):
        self.excludes = excludes
        self.root_dir = root_dir
        self.root = Node.from_directory(root_dir)
    
    # Will copy another context into the local
    # context
    def sync(self, other_context):
        self.excludes = other_context.excludes
    
    # In charge of launching a process within
    # this context
    def spawn(self, args):
        return subprocess.Popen(["poetry", "run"] + args, cwd=self.root_dir)
    
    # By default close() does nothing
    def close(self):
        pass

    # Context manager to clean up any context-dependent resources
    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    @staticmethod
    def auto_create():
        return Context(".", [
            "wandb", ".gitignore", ".venv", ".git", ".pytest_cache"]
        )
    
class TempContext(Context):
    # Create an empty temp context
    def __init__(self):
        directory = tempfile.mkdtemp(prefix="stanza_")
        super().__init__(directory, [])
    
    def close(self):
        # Delete temporary directory
        pass

# Represents the interface to a worker.
# Basically you can send it pickled functions
# and it will give you a reference to execute that function
class Worker:
    def __init__(self, id):
        self.id = id

    def create_executor(pickled_fun):
        # unpickle the function on this side of the network
        fun = cloudpickle.loads(pickled_fun)
        return fun

# All spawner services
# share a worker manager
class WorkerManager:
    def __init__(self, service_port):
        self.service_port = service_port
        # id --> 
        self.workers = {}
        self.id_events = {}
        self.ctx = TempContext()
    
    def register_worker(self, id, worker):
        self.workers[id] = worker
        # If someone is waiting for the 
        # worker to start, set the event
        if id in self.id_events:
            self.id_events[id].set()

    def spawn_workers(self, context, n):
        # Copy the context
        self.ctx.sync(context)

        ports = []

        startup_events = []
        ids = []
        for i in range(n):
            id = uuid.uuid4()
            event = Event()

            startup_events.append(event)
            ids.append(id)

            proc = context.spawn(
                ["python", "-m", "stanza.experiment.launch_worker",
                "--id", id, "--manager_port", f"{self.service_port}",
                "--root_dir", self.ctx.root_dir]
            )
        # Wait for all workers to have started
        for e in startup_events:
            e.wait()
        # get the workers by the ids
        workers = [self.workers[i] for i in ids]
        return workers
    
    def start(self):
        server = ThreadedServer(
            classpartial(WorkerSpawnService, manager),
            port=self.service_port
        )
        server.start()

class WorkerSpawnService(rpyc.Service):
    def __init__(self, manager):
        self.manager = manager
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
    
    def exposed_register_worker(self, id, worker):
        self.manager.register_worker(id, worker)

    def exposed_spawn_workers(self, context, n):
        workers = self.manager.spawn_workers(context, n)
        self.workers.extend(workers)
        return workers