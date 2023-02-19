from typing import Any

class AttrDict(dict):
    def __setattr__(self, name):
        return self[name]

    def __setattr__(self, name: str, value: Any):
        self[name] = value