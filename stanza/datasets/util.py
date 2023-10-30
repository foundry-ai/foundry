import requests
import os
from rich.progress import Progress
from pathlib import Path
import zipfile

_DATA = Path(os.environ["HOME"]) / ".dataset_cache"

def cache_path(key, filename=None):
    path = _DATA / key 
    return path / filename

def download_and_extract(
        download_path,
        extract_path,
        url=None,
        gdrive_id=None,
        strip_folder=False,
        quiet=False):
    download(download_path, url=url,
             gdrive_id=gdrive_id, quiet=quiet)
    extract_zip(download_path, extract_path, 
        strip_folder=strip_folder, quiet=quiet)

def extract_zip(path, dest, strip_folder=False, quiet=False):
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, 'r') as zip_ref:
        total_size_in_bytes = sum([zinfo.file_size for zinfo in zip_ref.infolist()])

        if quiet:
            for zipinfo in zip_ref.infolist():
                if strip_folder:
                    path = Path(zipinfo.filename)
                    path = Path(*path.parts[2:])
                    zipinfo.filename = str(path)
                zip_ref.extract(zipinfo, dest)
        else:
            with Progress() as pbar:
                task = pbar.add_task("Extracting...", total=total_size_in_bytes)
                for zipinfo in zip_ref.infolist():
                    if strip_folder:
                        path = Path(zipinfo.filename)
                        if len(path.parts) < 2:
                            continue
                        path = Path(*path.parts[1:])
                        zipinfo.filename = str(path)
                    zip_ref.extract(zipinfo, dest)
                    pbar.update(task, advance=zipinfo.file_size)

def download(path, url=None, gdrive_id=None, quiet=False):
    if path.exists():
        return path
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    if gdrive_id is not None:
        download_gdrive(path, gdrive_id, quiet=quiet)
        return path
    elif url is not None:
        download_file(path, url, quiet=quiet)
        return path

def download_file(path, url, quiet=False):
    if path.is_file():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024*10 #10 Kibibyte
    if quiet:
        with open(path, "wb") as f:
            for data in response.iter_content(block_size):
                f.write(data)
    else:
        with Progress() as pbar:
            task = pbar.add_task("Downloading...", total=total_size_in_bytes)
            with open(path, "wb") as f:
                for data in response.iter_content(block_size):
                    f.write(data)
                    pbar.update(task, advance=len(data))

def download_gdrive(path, id, quiet=False):
    import gdown
    with open(path, "wb") as f:
        gdown.download(id=id, output=f, quiet=quiet)