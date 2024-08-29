import requests
import re
import bs4
import urllib
import os
import hashlib
import shutil

import foundry.numpy as jnp
import numpy as np

from PIL import Image
from rich.progress import Progress
from pathlib import Path
from foundry import dataclasses
import zipfile
import requests
import tarfile

import logging
logger = logging.getLogger(__name__)

# Downloading and extracting utilities...
_DATA = Path(os.environ["HOME"]) / ".dataset_cache"

def cache_path(key, filename=None):
    path = _DATA / key 
    return path / filename if filename is not None else path

def md5_file(path):
    with open(path, "rb") as f:
        file_hash = hashlib.md5()
        while chunk := f.read(8192):
            file_hash.update(chunk)
        return file_hash.hexdigest()

def download_and_extract(
        download_path : Path,
        extract_path : Path,
        *,
        job_name=None,
        url=None,
        gdrive_id=None,
        strip_folder=False,
        quiet=False,
        md5=None
    ):
    download(download_path, job_name=job_name, url=url,
             gdrive_id=gdrive_id, md5=md5, quiet=quiet)
    extract_to(download_path, extract_path, job_name=job_name,
        strip_folder=strip_folder, quiet=quiet)
    
@dataclasses.dataclass
class ExtractInfo:
    filename: str
    size: int
    is_dir: bool

def extract(archive_path, handler, *, job_name=None, quiet=False):
    def do_extract(progress_cb=None):
        ext = "".join(archive_path.suffixes)
        if ext == ".zip":
            f = zipfile.ZipFile(archive_path, "r")
            with f as f:
                total_size_in_bytes = sum(
                    [zinfo.file_size for zinfo in f.infolist()]
                )
                for zipinfo in f.infolist():
                    info = ExtractInfo(
                        filename=zipinfo.filename.strip("/"),
                        size=zipinfo.file_size,
                        is_dir=zipinfo.is_dir()
                    )
                    file = f.open(zipinfo.filename) if not info.is_dir else None
                    handler(info, file)
                    progress_cb(zipinfo.file_size, total_size_in_bytes)
        elif ext == ".tar.gz" or ext == ".tgz" or ext == ".tar":
            f = tarfile.open(archive_path, "r")
            with f as f:
                members = f.getmembers()
                total_size_in_bytes = sum(
                    [tarinfo.size for tarinfo in f.getmembers()]
                )
                for tarinfo in members:
                    info = ExtractInfo(
                        filename=tarinfo.name.strip("/"),
                        size=tarinfo.size,
                        is_dir=tarinfo.isdir()
                    )
                    file = f.extractfile(tarinfo)
                    handler(info, file)
                    progress_cb(tarinfo.size, total_size_in_bytes)
        else:
            raise ValueError(f"Unsupported archive type {ext}")
    if quiet:
        do_extract(lambda prog, total: None)
    else:
        with Progress() as progress:
            task = progress.add_task("Extracting..." 
                if job_name is None else f"Extracting {job_name}"
            )
            def cb(prog, total):
                progress.update(task, total=total, advance=prog)
            do_extract(cb)

def extract_to(archive_path : Path, dest : Path, *,
               job_name=None, strip_folder=None, quiet=False):
    dest = Path(dest)
    def handler(info, file):
        filename = (
            info.filename[len(strip_folder):]
            if strip_folder and info.filename.startswith(strip_folder) 
            else info.filename
        ).lstrip("/")
        path = dest / filename
        if info.is_dir:
            path.mkdir(exist_ok=True, parents=True)
        else:
            with file as src, open(path, "wb") as tgt:
                shutil.copyfileobj(src, tgt)
    dest.mkdir(parents=True, exist_ok=True)
    extract(archive_path, handler, job_name=job_name, quiet=quiet)

# ----------------
# Download utility
# ----------------

def download(path, *, job_name=None,
             url=None, gdrive_id=None, md5=None, quiet=False):
    path = Path(path)
    if path.exists():
        return path
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    if gdrive_id is not None:
        _download_gdrive(path, gdrive_id, job_name=job_name, quiet=quiet)
    elif url is not None:
        _download_url(path, url, job_name=job_name, quiet=quiet)
    else:
        raise ValueError("Must provide either url or gdrive_id")
    actual_md5 = md5_file(path)
    if md5 is not None:
        if actual_md5 != md5:
            raise ValueError(f"MD5 mismatch for {path}, expected {md5} got {actual_md5}")
    else:
        logger.debug(f"MD5 for downloaded {path} is {actual_md5}")
    return path

def _make_session():
    user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)" \
                " AppleWebKit/537.36 (KHTML, like Gecko)" \
                " Chrome/121.0.0.0 Safari/537.36"
    sess = requests.session()
    sess.headers.update({"User-Agent": user_agent})
    return sess

def _parse_confirmation(html):
    m = re.search(r'href="(\/uc\?export=download[^"]+)', html)
    if m:
        url = "https://docs.google.com" + m.groups()[0]
        url = url.replace("&amp;", "&")
        return url
    m = re.search('"downloadUrl":"([^"]+)', html)
    soup = bs4.BeautifulSoup(html, features="html.parser")
    form = soup.select_one("#download-form")
    if form is not None:
        url = form["action"].replace("&amp;", "&")
        url_components = urllib.parse.urlsplit(url)
        query_params = urllib.parse.parse_qs(url_components.query)
        for param in form.findChildren("input", attrs={"type": "hidden"}):
            query_params[param["name"]] = param["value"]
        query = urllib.parse.urlencode(query_params, doseq=True)
        url = urllib.parse.urlunsplit(url_components._replace(query=query))
        return url
    if m:
        url = m.groups()[0]
        url = url.replace("\\u003d", "=")
        url = url.replace("\\u0026", "&")
        return url
    m = re.search('<p class="uc-error-subcaption">(.*)</p>', html)
    if m:
        error = m.groups()[0]
        raise IOError(error)
    raise IOError(
        "No public link. Make sure sharing permissions are correctly set."
    )

def _download_gdrive(path, id, job_name=None, quiet=False):
    sess = _make_session()
    url = f"https://drive.google.com/uc?export=download&id={id}"
    while True:
        res = sess.get(url, stream=True, verify=True)
        if "Content-Disposition" in res.headers:
            break
        try:
            url = _parse_confirmation(res.text)
        except IOError as e:
            raise IOError(
                f"Failed to parse the confirmation page for {id}:\n {e}"
            )
    _download_url(path, url, job_name=job_name, quiet=quiet, response=res)

def _download_url(path, url, job_name=None, quiet=False,
                  response=None):
    if path.is_file():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, stream=True) if response is None else response
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024*10 #10 Kibibyte
    if quiet:
        with open(path, "wb") as f:
            for data in response.iter_content(block_size):
                f.write(data)
    else:
        with Progress() as pbar:
            task = pbar.add_task(
                f"Downloading {job_name}" if job_name is not None else "Downloading", 
                total=total_size_in_bytes
            )
            with open(path, "wb") as f:
                for data in response.iter_content(block_size):
                    f.write(data)
                    pbar.update(task, advance=len(data))
