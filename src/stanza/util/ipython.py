from PIL import Image
from pathlib import Path
import numpy as np
import tempfile
import ffmpegio
import uuid

from ipywidgets import Video, Image, HBox, HTML

STYLE = "<style>" \
    ".cell-output-ipywidget-background {" \
    "   background-color: transparent !important;" \
    "}" \
    ".jp-OutputArea-output {" \
    "   background-color: transparent;" \
    "}" \
    "</style>"

def display_image(array):
    array = np.array(array)
    if array.dtype == np.float32 or array.dtype == np.float64:
        array = (array*255).astype(np.uint8)
    img = Image.fromarray(array)
    id = uuid.uuid4()
    path = Path("/tmp") / (str(id) + ".png")
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return HBox([HTML(STYLE), Image.from_file(path)])

def display_video(array, fps=28):
    array = np.array(array)
    if array.dtype == np.float32 or array.dtype == np.float64:
        array = (array*255).astype(np.uint8)
    f = tempfile.mktemp() + ".mp4"
    id = uuid.uuid4()
    path = Path("/tmp") / (str(id) + ".mp4")
    path.parent.mkdir(parents=True, exist_ok=True)
    ffmpegio.video.write(path, fps, array)
    return HBox([HTML(STYLE), Video.from_file(path)])