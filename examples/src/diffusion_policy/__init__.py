from stanza.struct import click
from stanza import struct

from common import TrainConfig

@struct.dataclass
class Config:
    train: TrainConfig = TrainConfig()

@click.command()
@click.option("--config", type=Config, prefix_fields=False)
def train(params):
    pass