from .worker import Worker
import os
import rpyc
import argparse

from stanza.logging import logger

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--id')
    parser.add_argument('--manager_port', type=int)
    args  = parser.parse_args()
    worker = Worker()
    # Connect to the spawner and report that we have arrived!
    logger.trace(f"Started worker [yellow]{args.id}[/yellow] in [blue]{os.getcwd()}[/blue]")
    conn = rpyc.connect("localhost", args.manager_port,
        config={
            'allow_public_attrs': True
        }
    )
    conn.root.register_worker(args.id, worker)
    # Serve events
    conn.serve_all()