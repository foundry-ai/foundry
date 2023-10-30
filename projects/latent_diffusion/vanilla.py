from stanza.dataclasses import dataclass
from stanza.runtime import activity
import stanza.datasets as datasets

@dataclass(jax=True)
class Config:
    dataset : str = "celeb_a"

@activity(Config)
def train(config, db):
    train, test = datasets.load(
        config.dataset,
        splits=("train", "test")
    )