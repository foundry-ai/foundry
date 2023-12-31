import os
from stanza.reporting import Backend, BucketBackend, Video, Figure
from pathlib import Path

import urllib.parse
import jax
import jax.numpy as jnp
import random
import pickle

_NOUNS = ["freedom", "shield", "resolve",
    "shark", "sky", "champion",
    "dawn", "fury",
    "thunder", "endeavour",
    "enterprise", "discovery",
    "star", "sunset", "eagle"]
_ADJECTIVES = ["enduring", "hopeful"
    "blue","red", "dark", "light",
    "brilliant", "radiant", "silver", "daring"
    "deep", "hot", "cold", "burning",
    "gold", "starry", "proximal", "gallant"
    "defiant", "beautiful", "respectful"]

class LocalBackend(Backend):
    def __init__(self, url):
        parsed = urllib.parse.urlparse(url)
        path = parsed.path
        if not path:
            path = "/tmp/stanza"
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)

    @property
    def buckets(self):
        runs = list(self.path.iterdir())
        return LocalBuckets(runs)

    def find(self, id=None, **tags):
        buckets = list(
            [x for x in self.buckets \
             if (id is None or x.id == id) and x.has_tags(tags)]
        )
        return buckets

    def open(self, id):
        return LocalBucket(self.path / id)
    
    def create(self):
        num = len(list(self.path.iterdir())) + 1
        adj = random.choice(_ADJECTIVES)
        noun = random.choice(_NOUNS)
        name = f"{adj}-{noun}-{num}"
        path = self.path / name
        path.mkdir(exist_ok=True)
        return LocalBucket(path)

class LocalBuckets:
    def __init__(self, paths):
        self._paths = paths

    @property
    def latest(self):
        runs = sorted(self, key=lambda r: r.creation_time, reverse=True)
        if len(runs) == 0:
            return None
        return runs[0]
    
    def __len__(self):
        return len(self._paths)
    
    def __iter__(self):
        for p in self._paths:
            return LocalBucket(p)


class LocalBucket(BucketBackend):
    def __init__(self, path : Path):
        self._path = path
        self.step = 0
    
    @property
    def id(self):
        return self._path.name
    
    @property
    def url(self):
        return f"local://{self._path}"
    
    @property
    def tags(self):
        tags = [x.name.split(":")[1:] for x in self._path.iterdir() if x.name.startswith("tag:")]
        return {k: v for (k,v) in tags }
    
    def tag(self, **tags):
        tags = ( f"{k}:{v}" for (k,e) in tags.items() for v in e )
        for n in tags:
            path = self._path / f"tag:{n}"
            path.touch()
    
    def add(self, name, value, *,
            append=False, step=None,
            batch=False):
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

# class LocalDatabase(Database):
#     def __init__(self, *, parent=None, name=None, path=None):
#         self._name = name
#         self._parent = parent
#         if path is None:
#             path = Path(os.getcwd()) / "results"
#         self._path = Path(path)
#         self._path.mkdir(parents=True, exist_ok=True)
    
#     @property
#     def name(self):
#         return self._name

#     @property
#     def parent(self):
#         return self._parent

#     @property
#     def children(self):
#         return set([p.stem for p in self._path.iterdir()])

#     def has(self, name):
#         return name in self.children

#     def open(self, name=None):
#         if name is None:
#             length = len(self.children)
#             while True:
#                 adjectives, nouns = _words()
#                 adjective = random.choice(adjectives)
#                 noun = random.choice(nouns)
#                 name = f"{adjective}-{noun}-{length + 1}"
#                 if not self.has(name):
#                     break
#         return LocalDatabase(parent=self, name=name, path=self._path / name)

#     def add(self, name, value, *, append=False, step=None, batch=False):
