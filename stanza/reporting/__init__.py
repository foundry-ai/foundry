import wandb
import numpy as np

class Image(wandb.Image):
    def __init__(self, data, *args, 
            display_width=None,
            display_height=None, **kwargs):
        data = np.array(data)
        super().__init__(data, *args, **kwargs)
        self._display_width = display_width
        self._display_height = display_height
    
    def _repr_png_(self):
        img = self._image
        if not img and self._path:
            with open(self._path, 'rb') as f:
                return f.read()
        return img.tobytes()