import IPython.display as display
import numpy as np
import io
import ffmpegio
from PIL import Image
from pathlib import Path

def display_image(image):
    img = Image.fromarray(np.array((255*image).astype(np.uint8)), 'RGB')
    imgByteArr = io.BytesIO()
    # image.save expects a file-like as a argument
    img.save(imgByteArr, format='PNG')
    # Turn the BytesIO object back into a bytes object
    imgByteArr = imgByteArr.getvalue()
    return display.Image(data=imgByteArr)

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