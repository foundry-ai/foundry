import IPython.display as display
import numpy as np
import math
import io
import ffmpegio
from PIL import Image
from pathlib import Path

from stanza.data import Data, PyTreeData
from stanza.util import make_grid

import jax
import jax.numpy as jnp

def display_image(image):
    if image.dtype != np.uint8:
        image = np.array((255*image)).astype(np.uint8)
    else:
        image = np.array(image)
    img = Image.fromarray(image)
    imgByteArr = io.BytesIO()
    # image.save expects a file-like as a argument
    img.save(imgByteArr, format='PNG')
    # Turn the BytesIO object back into a bytes object
    imgByteArr = imgByteArr.getvalue()
    return display.Image(data=imgByteArr)

def display_images(images, cols=None, rows=None):
    if isinstance(images, Data):
        images = PyTreeData.from_data(images, chunk_size=64).data
    image = make_grid(images, cols, rows)
    return display_image(image)

def display_html(html):
    return display.HTML(html)

def display_video(video_data=None, fps=30):
    path = Path("/tmp/notebook_videos") / "foo.mp4"
    path.parent.mkdir(parents=True, exist_ok=True)
    path = path.with_suffix(".mp4")
    if video_data is not None:
        if video_data.dtype == np.float32:
            video_data = (255*video_data).astype(np.uint8)
        ffmpegio.video.write(path, fps, video_data,
                    overwrite=True, loglevel="quiet")
    return display.Video(str(path), embed=True)