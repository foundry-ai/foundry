import os
from stanza.runtime.database import Database, Video, Figure
from pathlib import Path
import jax.numpy as jnp

class LocalDatabase(Database):
    def __init__(self, parent=None, name=None, path=None):
        self._name = name
        self._parent = parent
        if path is None:
            path = Path(os.getcwd()) / "results"
        self._path = Path(path)
        self._path.mkdir(parents=True, exist_ok=True)
    
    @property
    def name(self):
        return self._name

    @property
    def parent(self):
        return self._parent
    @property
    def children(self):
        return set([os.path.splitext(p)[0] for p in self._path.iterdir()])
    
    def has(self, name):
        path = self._path / name
        return path.exists()

    def open(self, name):
        return LocalDatabase(self, name, self._path / name)
    
    def add(self, name, value, append=False):
        path = self._path / name
        if path.is_dir():
            raise RuntimeError("This is a sub-database!")
        if isinstance(value, Video):
            import ffmpegio
            path = self._path / f"{name}.mp4"
            ffmpegio.video.write(path, value.fps, value.data,
                overwrite=True, loglevel='quiet')
        elif isinstance(value, Figure):
            path = self._path / f"{name}.npy"
            with open(path, "wb") as f:
                jnp.save(f, value, allow_pickle=True)

    def get(self, name):
        children = self.children
        if name not in children:
            return None
        matches = list(filter(
            lambda x: os.path.splitext(x)[0] == name,
            self._path.iterdir()
        ))
        if not matches:
            return None
        path = self._path / matches[0]
        with open(path, "rb") as f:
            arr = jnp.load(f, allow_pickle=True)
            if arr.dtype == object:
                return arr.item()
            return arr