from stanza.dataclasses import dataclass, field

_STORAGES = {}

class StoredData:
    length: int = field(default=0)

    @property
    def start(self):
        pass