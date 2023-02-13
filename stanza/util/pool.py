import queue
import os

import socket
import asyncio
import cloudpickle
import rpyc
import threading
import functools

from stanza.logging import logger
from .container import Image, Target
from rpyc.utils.server import ThreadedServer

def worker_id():
    return int(os.environ['REPLICA']) if 'REPLICA' in os.environ else None

# The remote worker object
class WorkerRemote:
    def __init__(self, id, loop):
        self.exposed_id = id
        self.loop = loop

    def exposed_create_executor(self, pickled_function):
        fun = cloudpickle.loads(pickled_function)
        def fun_wrapper(args, kwargs):
            # unpickle args, kwargs
            args = cloudpickle.loads(args)
            kwargs = cloudpickle.loads(kwargs)
            res = fun(*args, **kwargs)
            # pickle the result
            return cloudpickle.dumps(res)
        return fun_wrapper

class WorkerClient:
    def __init__(self, remote):
        self.remote = remote
        self.id = remote.id
    
    # will do the reverse pickling steps of the service
    def create_executor(self, function):
        fun = cloudpickle.dumps(function)
        remote_fun = self.remote.create_executor(fun)

        async def fun_unwrapper(*args, **kwargs):
            # launch the async call
            args = cloudpickle.dumps(args)
            kwargs = cloudpickle.dumps(kwargs)

            loop = asyncio.get_event_loop()
            done = asyncio.Event()

            res = rpyc.async_(remote_fun)(args, kwargs)
            res.add_callback(lambda _: loop.call_soon_threadsafe(done.set))

            await done.wait()
            res = res.value # grab result the value
            return cloudpickle.loads(res)
        return fun_unwrapper

class PoolService(rpyc.Service):
    def __init__(self, pool):
        self.pool = pool
    
    def exposed_register_worker(self, remote_worker):
        self.pool.register_worker(WorkerClient(remote_worker))



# For the "Pool" you specify a docker service # onto which to deploy the pool
class Pool:
    def __init__(self, target):
        if isinstance(target, str):
            self.target = Target.from_url(target)
        else:
            self.target = target
        self.image = Image.current()

        # use port 0 to pick a free port
        self.server = ThreadedServer(PoolService(self), port=0)
        self.server_thread = threading.Thread(target=self.server.start, daemon=True)

        self.containers = None
        self.workers = []
        self.expected_workers = self.target.num_replicas

        self._init_loop = None
        self._workers_initialized = asyncio.Event()
    
    # callback from the pool service
    def register_worker(self, worker):
        #logger.trace("pool", f"Registering remote worker {worker.id}")
        self.workers.append(worker)
        # check if we should signal all workers online
        if len(self.workers) >= self.expected_workers:
            # happens in a different thread
            self._init_loop.call_soon_threadsafe(self._workers_initialized.set)
    
    async def init(self):
        if not self.containers:
            self._init_loop = asyncio.get_event_loop()

            logger.info(f"Starting RPC server on {self.server.host}:{self.server.port}")
            self.server_thread.start()

            # load the image into the target engine
            logger.info("pool", "Ingesting image into target engine.")
            img = await self.target.engine.ingest(self.image)
            logger.info("pool", "Spawning containers")
            self.containers = await self.target.launch(img,
                                    ["python", "-m", "stanza.util.launch_worker"], 
                                    env={'RPC_HOST': socket.gethostname(),
                                        'RPC_PORT': self.server.port})
            # wait for all workers to have connected
            logger.info("pool", "Waiting for workers")
            await self._workers_initialized.wait()
            logger.info("pool", "All workers connected")
    
    async def close(self):
        # kill all the containers
        #await asyncio.gather(*[c.stop() for c in self.containers])
        pass

    async def run(self, func, iterable):
        executors = asyncio.Queue()
        for w in self.workers:
            executors.put_nowait(w.create_executor(func))

        tasks = set()
        for i in iterable:
            e = await executors.get()
            task = asyncio.create_task(e(i))
            # add the executor back to the executors
            def done(e, _):
                executors.put_nowait(e)
            task.add_done_callback(functools.partial(done, e))
            tasks.add(task)
        # wait for all launched tasks to finish
        # TODO: Remove a task form tasks when done
        # right now a task remains in tasks even if done
        await asyncio.gather(*tasks)

    async def __aenter__(self):
        await self.init()
        return self
    
    async def __aexit__(self, exc_type, exc, tb):
        await self.close()

class ProcPool:
    def __init__(self):
        pass

class DockerPool:
    def __init__(self, group):
        pass