from stanza.reporting import Backend, Bucket
import urllib.parse as urlparse

import itertools

class DummyBackend(Backend):
    def __init__(self, url=None):
        self.url = url or "dummy://"
        self.buckets = []

    def open(self, id):
        return self.buckets[int(id)]

    def create(self):
        id = str(len(self.buckets))
        url = self.url + "/" + id
        bucket = DummyBucket(url, id)
        self.buckets.append(bucket)
        return bucket

class DummyBucket:
    def __init__(self, url, id):
        self.url = url
        self.id = id
        self.tags = {}
    
    def tag(self, **tags):
        for (k,v) in tags.items():
            self.tags.setdefault(k, set()).update(v)

    def add(self, name, value, *,
            append=False, step=None,
            batch=False, batch_lim=None):
        pass

    def get(self, name):
        pass
    
    def log(self, data, step=None, batch=False):
        pass