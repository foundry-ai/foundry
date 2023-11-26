from stanza.datasets import builder

from stanza.data.stored import FolderImageStorage
from stanza.data.stored import StoredData

from .util import cache_path, download_and_extract

@builder
def celeb_a(quiet=False, splits=set()):
    download_path = cache_path("celeb_a", "img_align_celeba.zip")
    folder_path = cache_path("celeb_a", "data")
    if not folder_path.exists():
        download_and_extract(
            download_path=download_path,
            extract_path=folder_path,
            gdrive_id="1Yo6KZFeQeuplQ_fvqvqAei0WouFbjKjT",
            quiet=quiet,
            strip_folder=True
        )
    data = {}
    val_start = 162772
    test_start = 182638
    if "train" in splits:
        storage = FolderImageStorage(folder_path, end=val_start)
        data["train"] = StoredData(storage)
    if "validation" in splits:
        storage = FolderImageStorage(folder_path,
                    start=val_start, end=test_start)
        data["validation"] = StoredData(storage)
    if "test" in splits:
        storage = FolderImageStorage(
            folder_path,
            start=test_start
        )
        data["test"] = StoredData(storage)
    return data