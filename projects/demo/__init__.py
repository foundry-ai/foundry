from stanza.runtime import activity
from stanza.util.dataclasses import dataclass
from stanza.util.logging import logger

@dataclass
class Config:
    seed: int = 42
    name: str = "foo"

@activity(Config)
def demo(config, database):
    logger.info("Parsed config: {}", config)