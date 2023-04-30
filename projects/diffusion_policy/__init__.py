from stanza.runtime import activity
from stanza.util.dataclasses import dataclass, field

import stanza.envs as envs

@dataclass
class Config:
    env: str = "pusht"

def create_expert_data(env_type):
    if env_type == "pusht":
        from stanza.envs.pusht import expert_data
        return expert_data

@activity
def train_policy(config, database):
    env = envs.create(config.env)
    data_factory = datafactory()