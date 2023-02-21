from typing import Any

class AttrDict(dict):
    def __setattr__(self, name: str, value: Any):
        self[name] = value
    
    def __getattr__(self, name: str):
        return self[name]