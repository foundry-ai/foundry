import requests
from rich.progress import Progress

def _download(url, path, quiet=False):
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

