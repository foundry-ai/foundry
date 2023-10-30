from stanza.datasets import builder

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
    if "train" in splits:
        data["train"] = None
    if "test" in splits:
        data["test"] = None
    return data