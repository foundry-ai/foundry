import requests
import os
from rich.progress import Progress
from pathlib import Path
import zipfile
import tarfile

_DATA = Path(os.environ["HOME"]) / ".dataset_cache"

def cache_path(key, filename=None):
    path = _DATA / key 
    return path / filename if filename is not None else path

def download_and_extract(
        download_path : Path,
        extract_path : Path,
        url=None,
        gdrive_id=None,
        strip_folder=False,
        quiet=False):

    download(download_path, url=url,
             gdrive_id=gdrive_id, quiet=quiet)
    extract(download_path, extract_path, 
        strip_folder=strip_folder, quiet=quiet)

def extract(archive_path : Path, dest : Path, strip_folder=False, quiet=False):
    dest.mkdir(parents=True, exist_ok=True)
    def do_extract(progress_cb=None):
        ext = "".join(archive_path.suffixes)
        if ext == ".zip":
            f = zipfile.ZipFile(archive_path, "r")
        elif ext == ".tar.gz":
            f = tarfile.open(archive_path, "r")
        else:
            raise ValueError(f"Unknown archive type: {archive_path.suffix}")
        with f as f:
            if ext == ".zip":
                total_size_in_bytes = sum(
                    [zinfo.file_size for zinfo in f.infolist()]
                )
                for zipinfo in f.infolist():
                    if strip_folder:
                        path = Path(zipinfo.filename)
                        path = Path(*path.parts[2:])
                        zipinfo.filename = str(path)
                    f.extract(zipinfo, dest)
                    progress_cb(zipinfo.file_size, total_size_in_bytes)
            else:
                f.extractall(dest)
    if quiet:
        do_extract()
    else:
        with Progress() as progress:
            task = progress.add_task("Extracting...")
            def cb(prog, total):
                progress.update(task, total=total, advance=prog)
            do_extract(cb)

def download(path, url=None, gdrive_id=None, quiet=False):
    path = Path(path)
    if path.exists():
        return path
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    if gdrive_id is not None:
        # download_gdrive(path, gdrive_id, quiet=quiet)
        print("warning gdrive not implemented")
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

# def download_gdrive(path, id, quiet=False):
#     import gdown
#     with open(path, "wb") as f:
#         gdown.download(id=id, output=f, quiet=quiet)