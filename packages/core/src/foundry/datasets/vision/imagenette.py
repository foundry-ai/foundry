from foundry.datasets import DatasetRegistry
from foundry.datasets.vision import ImageClassDataset

from ..util import cache_path, download

import functools

URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-{res}.tgz"
HASHES = {
    "160": "e793b78cc4c9e9a4ccc0c1155377a412"
}

def _load_raw(res, quiet=False):
    data_path = cache_path(f"imagenette_{res}")
    file_path = data_path / "data.tgz"
    url = URL.format(res=res)
    download(file_path, url=url,
             job_name=f"Imagenette {res}px", quiet=quiet,
             md5=HASHES[res])

def _load(res, quiet=False, **kwargs):
    _load_raw(res, quiet=quiet)

datasets = DatasetRegistry()
datasets.register("160px", functools.partial(_load, res="160"))
datasets.register("320px", functools.partial(_load, res="320"))