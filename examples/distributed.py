from stanza.experiment.worker import Context, WorkerPool, PoolInfo
from stanza.logging import logger
import sys

data = range(20)

host = sys.argv[1] if len(sys.argv) > 1 else None


with WorkerPool(Context.default(), PoolInfo(5, host)) as p:
    res = p.map(lambda x: x*x, data)

logger.info(f"{res}")