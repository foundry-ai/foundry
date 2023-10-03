import wandb
import jax
import dateutil.parser

from stanza.reporting import Backend, Video, Figure
import numpy as np
import os
import pickle

import urllib.parse

class WandbBackend(Backend):
    def __init__(self, url):
        parsed = urllib.parse.urlparse(url)
        entity = parsed.netloc
        path = parsed.path.split("/")
        project = path[1] if len(path) > 1 else "default"
        self.api = wandb.Api()
        self.entity = entity
        self.project = project
    
    @property
    def buckets(self):
        runs = self.api.runs(path=self.entity + "/" + self.project)
        return WandbRuns(runs)

    def find(self, id=None, **tags):
        # convert the tags to a list of strings
        tags = list([ f"{k}:{v}" for (k,e) in tags.items() for v in e ])
        runs = self.api.runs(
            path=self.entity + "/" + self.project,
            filters={"tags": {"$in": tags}}
        )
        return WandbRuns(runs)

    def open(self, id):
        run = wandb.init(id=id)
        return WandbRun(run)

    def create(self):
        if "WANDB_RUN_ID" in os.environ:
            run = wandb.init()
        else:
            run = wandb.init(
                entity=self.entity, project=self.project)
        return WandbRun(run)

class WandbRuns:
    def __init__(self, runs):
        self._runs = runs
    
    @property
    def latest(self):
        runs = sorted(self, key=lambda r: r.creation_time, reverse=True)
        if len(runs) == 0:
            return None
        return runs[0]
    
    def __len__(self):
        return len(self._runs)
    
    def __iter__(self):
        for r in self._runs:
            yield WandbRun(r)

class WandbRun:
    def __init__(self, run):
        self._run = run
    
    @property
    def creation_time(self):
        return dateutil.parser.isoparse(self._run.created_at)
    
    @property
    def id(self):
        return self._run.id
    
    @property
    def url(self):
        return f"wandb://{self._run.entity}/{self._run.project}/{self._run.id}"
    
    @property
    def tags(self):
        pass

    def tag(self, **tags):
        tags = tuple(( f"{k}:{v}" for (k,e) in tags.items() for v in e ))
        self._run.tags = self._run.tags + tags

    def add(self, name, value, *,
            append=False, step=None,
            batch=False, batch_lim=None):
        if step is not None:
            raise RuntimeError("Cannot add with steps with wandb backend!")
        artifact = wandb.Artifact(name, type="")
        with artifact.new_file("data", mode="wb") as f:
            pickle.dump(value, f)
        self._run.log_artifact(artifact)

    def get(self, name):
        artifacts = self._run.logged_artifacts()
        for a in artifacts:
            aname = a.name.split(":")[0]
            if aname == name:
                file = a.get_path("data").download()
                with open(file, "rb") as f:
                    return pickle.load(f)

    def log(self, data, step=None, batch=False):
        if batch:
            dim = jax.tree_util.tree_leaves(data)[0].shape[0]
            for i in range(dim):
                x = jax.tree_map(lambda x: x[i], data)
                s = step[i] if step is not None else None
                self.log(x, step=s, batch=False)
        else:
            data = remap(data, {
                    Figure: lambda f: f.fig,
                    Video: lambda v: wandb.Video(np.array(v.data), fps=v.fps)
                })
            if self.prefix != '':
                data = {self.prefix: data}
            self.run.log(data, step=step)

def remap(obj, type_mapping):
    if isinstance(obj, dict):
        return { k: remap(v, type_mapping) for (k,v) in obj.items() }
    elif isinstance(obj, list):
        return [ remap(v, type_mapping) for v in obj ]
    elif isinstance(obj, tuple):
        return tuple([ remap(v, type_mapping) for v in obj ])
    elif isinstance(obj, np.ndarray):
        return np.array(obj)
    elif type(obj) in type_mapping:
        return type_mapping[type(obj)](obj)
    else:
        return obj