import os
from stanza.reporting import Database, Video, Figure
from pathlib import Path
import jax
import jax.numpy as jnp
import random
import pickle

_NOUNS = None
_ADJECTIVES = None
def _words():
    global _NOUNS
    global _ADJECTIVES
    if _NOUNS is None:
        nouns_path = Path(__file__).parent / "nouns.txt"
        _NOUNS = open(nouns_path).read().splitlines()
    if _ADJECTIVES is None:
        adjectives_path = Path(__file__).parent / "adjectives.txt"
        _ADJECTIVES = open(adjectives_path).read().splitlines()
    return _ADJECTIVES, _NOUNS

class LocalDatabase(Database):
    def __init__(self, *, parent=None, name=None, path=None):
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
        return set([p.stem for p in self._path.iterdir()])

    def has(self, name):
        return name in self.children

    def open(self, name=None):
        if name is None:
            length = len(self.children)
            while True:
                adjectives, nouns = _words()
                adjective = random.choice(adjectives)
                noun = random.choice(nouns)
                name = f"{adjective}-{noun}-{length + 1}"
                if not self.has(name):
                    break
        return LocalDatabase(parent=self, name=name, path=self._path / name)

    def add(self, name, value, *, append=False, step=None, batch=False):
        path = self._path / name
        if path.is_dir():
            raise RuntimeError("This is a sub-database!")
        if isinstance(value, Video):
            assert append == False
            import ffmpegio
            path = self._path / f"{name}.mp4"
            data = value.data
            if data.dtype == jnp.float32:
                data = (255*data).astype(jnp.uint8)
            if data.shape[-1] == 4:
                data = data[...,:3]
            #data = jnp.transpose(data, (0, 3, 1, 2))
            ffmpegio.video.write(path, value.fps, data,
                overwrite=True, loglevel='quiet')
        elif isinstance(value, Figure):
            assert append == False
            png_path = self._path / f"{name}.png"
            pdf_path = self._path / f"{name}.pdf"
            from plotly.graph_objects import Figure as GoFigure
            if isinstance(value.fig, GoFigure):
                value.fig.write_image(png_path)
                value.fig.write_image(pdf_path)
            else:
                if value.width:
                    value.fig.set_figwidth(value.width)
                if value.height:
                    value.fig.set_figheight(value.height)
                value.fig.savefig(png_path, bbox_inches='tight')
                value.fig.savefig(pdf_path, bbox_inches='tight')
        else:
            path = self._path / f"{name}.pkl"
            if append and path.is_file():
                with open(path, "rb") as f:
                    d = pickle.load(f)
                    value = jnp.expand_dims(value, 0) \
                        if not batch else value
                    value = jnp.concatenate(
                        (d, value),
                        axis=0
                    )
            with open(path, "wb") as f:
                pickle.dump(value, f)

    def get(self, name):
        children = self.children
        if name not in children:
            raise AttributeError()
        matches = list(filter(
            lambda x: x.stem == name, self._path.iterdir()
        ))
        if not matches:
            raise AttributeError()
        path = matches[0]
        with open(path, "rb") as f:
            return pickle.load(f)