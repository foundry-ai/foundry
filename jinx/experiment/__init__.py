import numpy as np
from loguru import logger

# TODO: No idea what this API should look like yet

# Type hint annotations
class Scalar:
    def log(self, value):
        self._log(np.array(value).item())
    
    def _log(self, value):
        pass


class Figure:
    def log(self, value):
        self._log(value)
    
    def _log(self, value):
        pass


class Lab:
    def create(self, name=None):
        pass

class Experiment:
    pass