from stanza.datasets import DatasetRegistry, ImageDataset
from stanza.data import Data, io
from stanza.data.normalizer import ImageNormalizer

import jax
import jax.numpy as jnp

from . import util as du

class CelebAData(Data):
    def __init__(self, path):
        self._path = path
    
    def __getitem__(self, idx):
        return io.read_image(
            self._path / "{:06d}.jpg", 
            (218, 178, 3),
            idx + 1
        )

    def __len__(self):
        return 202599

def _load_celeb_a(quiet=False, **kwargs):
    data_path = du.cache_path("celeb_a") / "img_align_celeba.zip"
    du.download(data_path,
        gdrive_id="1Yo6KZFeQeuplQ_fvqvqAei0WouFbjKjT",
        md5="00d2c5bc6d35e252742224ab0c1e8fcb",
        job_name="CelebA"
    )
    extract_path = du.cache_path("celeb_a") / "images"
    if not extract_path.exists():
        du.extract_to(data_path, extract_path,
            job_name="CelebA",
            quiet=quiet,
            strip_folder="img_align_celeba"
        )
    
    train_data = CelebAData(extract_path)

    return ImageDataset(
        splits={"train": train_data},
        normalizers={
            "hypercube": lambda: ImageNormalizer(jax.ShapeDtypeStruct((218, 178, 3), jnp.uint8)), 
        },
        transforms={}
    )


registry = DatasetRegistry()
registry.register("celeb_a", _load_celeb_a)