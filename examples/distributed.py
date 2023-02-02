from stanza.experiment import Context, WorkerPool, PoolInfo

data = range(20)

def func():
    pass
with WorkerPool(Context.auto_create(), PoolInfo(5)) as p:
    res = p.map(lambda x: x*x, data)
print(res)