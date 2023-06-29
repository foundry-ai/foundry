import IPython.display as display
import numpy as np
import io
import ffmpegio
from PIL import Image

def display_image(image):
    img = Image.fromarray(np.array((255*image).astype(np.uint8)), 'RGB')
    imgByteArr = io.BytesIO()
    # image.save expects a file-like as a argument
    img.save(imgByteArr, format='PNG')
    # Turn the BytesIO object back into a bytes object
    imgByteArr = imgByteArr.getvalue()
    return display.Image(data=imgByteArr)


def display_video(file_name, video_data=None, fps=30):
    if video_data is not None:
        if video_data.dtype == np.float32:
            video_data = (255*video_data).astype(np.uint8)
        ffmpegio.video.write(file_name, fps, video_data,
                    overwrite=True, loglevel='quiet')
    return display.Video(file_name)