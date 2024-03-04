from stanza.struct import args
from stanza import struct

from common import TrainConfig

@struct.dataclass
class Config:
    train: TrainConfig = TrainConfig()

@args.command(Config, "train")
def train(config):
    print(config)