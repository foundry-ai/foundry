from stanza.logging import logger

from .host import WorkerManager

def run():
    logger.info("Starting worker manager!")
    manager = WorkerManager(18860)
    manager.start()

if __name__ == "__main__":
    run()