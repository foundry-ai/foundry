import foundry.numpy as jnp
from foundry.core.dataclasses import dataclass
from foundry.data import Data, io
from . import Image, ImageDataset

from foundry.datasets.core import DatasetRegistry
from foundry.datasets import util

from pathlib import Path

@dataclass
class FFHQData(Data):
    path: Path
    start: int
    end: int

    def __getitem__(self, idx) -> Image:
        return Image(io.read_image(
            self.path / "{:05d}.png", 
            (128, 128, 3),
            idx + self.start
        ))
    
    def __len__(self):
        return self.end - self.start

@dataclass
class FFHQDataset(ImageDataset):
    path: Path

    def split(self, name):
        if name == "train":
            return FFHQData(self.path, 0, 50000)
        elif name == "test":
            return FFHQData(self.path, 50000, 60000)
        elif name == "validation":
            return FFHQData(self.path, 60000, 70000)

def _load_ffhq(quiet=False):
    extract_path = util.cache_path("ffhq") / "images"
    if not extract_path.exists():
        data_path = util.cache_path("ffhq") / "ffhq.zip"
        util.download(data_path,
            gdrive_id="1yKafRWamMCPojb5GBfqIPTgRoNEIISX6",
            md5="cf783275055f3f45a030101b6db4a3be",
            job_name="FFHQ"
        )
        util.extract_to(data_path, extract_path,
            job_name="FFHQ",
            quiet=quiet, strip_folder="thumbnails128x128"
        )
        data_path.unlink()
    return FFHQDataset(path=extract_path)

def register(registry: DatasetRegistry, prefix=None):
    registry.register("ffhq", _load_ffhq, prefix=prefix)