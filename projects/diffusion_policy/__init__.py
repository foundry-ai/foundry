from stanza.runtime import activity
from stanza.util.dataclasses import dataclass, field

import stanza.envs as envs

@dataclass
class Config:
    env: str = "pusht"

@activity
def train_policy(config, database):
    env = envs.create(config.env)