import IPython.display as display
import numpy as np
import io
from PIL import Image

def display_image(image):
    img = Image.fromarray(np.array((255*image).astype(np.uint8)), 'RGB')
    imgByteArr = io.BytesIO()
    # image.save expects a file-like as a argument
    img.save(imgByteArr, format='PNG')
    # Turn the BytesIO object back into a bytes object
    imgByteArr = imgByteArr.getvalue()
    return display.Image(data=imgByteArr)