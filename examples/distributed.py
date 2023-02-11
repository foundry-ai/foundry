import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import sys
from stanza.util.pool import Pool, worker_id
from stanza.logging import logger
import asyncio
import argparse
import stanza.util.pool

def func(x):
    logger.info(f"Worker {worker_id()}: got {x}")

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("target", default="poetry://localhost?n=5")
    args = parser.parse_args()

    data = range(20)
    async with Pool(args.target) as p:
        await p.run(func, data)

if __name__=="__main__":
    loop = asyncio.new_event_loop()
    loop.run_until_complete(main())
    loop.close()