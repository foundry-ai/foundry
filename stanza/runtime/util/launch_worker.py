import os
import sys
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
sys.path.append(os.path.abspath(os.path.join(__file__, "..","..","..", "..", "projects")))

from stanza.util.logging import logger
from stanza.runtime.pool import WorkerRemote
import argparse
import asyncio
import traceback
import time

# Change sigterm to give keyboardinterrupt
# so that docker can exit cleanly
import signal
def handle_sigterm(*args):
    raise KeyboardInterrupt()
signal.signal(signal.SIGTERM, handle_sigterm)

import rpyc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", default=os.environ.get("REPLICA", None), required=False)
    parser.add_argument("--host", default=os.environ.get("RPC_HOST", None))
    parser.add_argument("--port", type=int, default=int(os.environ.get("RPC_PORT", "18861")))
    parser.add_argument("--verbose", default=bool(os.environ.get("VERBOSE", False)), required=False)
    args = parser.parse_args()

    if args.verbose:
        logger.trace(f"Worker {args.id} connecting to [green]{args.host}[/green]:[cyan]{args.port}[/cyan]",
                     highlight=False)

    try:
        conn = rpyc.connect(args.host, args.port)
    except Exception:
        logger.error("{}", traceback.format_exc())
        while True:
            time.sleep(100)

    worker = WorkerRemote(args.id, asyncio.new_event_loop())
    conn.root.register_worker(worker)
    try:
        conn.serve_all()
    except KeyboardInterrupt:
        pass

if __name__=="__main__":
    main()