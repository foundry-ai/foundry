from stanza.runtime import activity
from stanza.util.dataclasses import dataclass
from stanza.util.logging import logger

import time

@dataclass(frozen=True)
class Config:
    seed: int = 42
    param: float = 0.1
    name: str = "foo"

@activity(Config)
def demo(config, database):
    logger.info("Parsed config: {}", config)
    time.sleep(5)
    logger.info("Done!")