import queue

class WorkerService:
    def __init__(self):
        pass

class WorkerClient:
    def __init__(self):
        pass

class ExecutorHandle:
    def __init__(self):
        pass

# For the "Pool" you specify a docker service
# onto which to deploy the pool
class Pool:
    def __init__(self):
        self.workers = []
    
    def run(self, func, iterable):
        executors = queue.Queue()
        for w in self.workers:
            executors.put(w.create_executor(func))

        v = []
        for i in iterable:
            e = executors.get()
            v.append(e(i))

    @staticmethod
    def from_env():
        pass

class ProcPool:
    def __init__(self):
        pass

class DockerPool:
    def __init__(self, group):
        pass