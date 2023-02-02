from .host import Worker
import rpyc

if __name__ == "__main__":
    parser = argparse.ArgmuentParser()
    parser.add_argument('--id')
    parser.add_argument('--manager_port', type=int)
    parser.add_argument('--root_dir')
    args  = parser.parse_args()

    worker = Worker()
    # Connect to the spawner and report that we have arrived!
    logger.info("Worker started.")
    conn = rpyc.connect("localhost", args.manager_port)
    conn.root.register_worker(args.id, worker)
    # Serve events
    logger.info("Waiting for requests...")
    conn.serve_all()