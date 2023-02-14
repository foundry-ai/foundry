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

def replica_id():
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

        self._init_loop = None

        self.workers_lock = asyncio.Condition()
        self.workers = []
    
    async def _register_callback(self, worker):
        async with self.workers_lock:
            self.workers.append(worker)
            self.workers_lock.notify_all()

    # callback from the pool service
    def register_worker(self, worker):
        self._init_loop.call_soon_threadsafe(
            lambda: self._init_loop.create_task(self._register_callback(worker))
        )
    
    async def init(self):
        if not self.containers:
            self.target = await self.target
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
    
    async def close(self):
        # kill all the containers
        await asyncio.gather(*[c.stop() for c in self.containers])

    async def run(self, func, iterable):
        executors = asyncio.Queue()
        async def populate_executors():
            async with self.workers_lock:
                num_executors = 0
                while True:
                    for w in self.workers[num_executors:]:
                        executors.put_nowait(w.create_executor(func))
                        num_executors = num_executors + 1
                    await self.workers_lock.wait()
        async def run_tasks():
            tasks = set()
            for i in iterable:
                e = await executors.get()
                task = asyncio.create_task(e(i))
                # add the executor back to the executors
                def done(e, _):
                    executors.put_nowait(e)
                task.add_done_callback(functools.partial(done, e))
                tasks.add(task)
            await asyncio.gather(*tasks)
        # run populate_executors and run_tasks simulatenously
        # until run_tasks is done, at which point cancel populate_executors
        try:
            populate_task = asyncio.create_task(populate_executors())
            await run_tasks()
        finally:
            populate_task.cancel()

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