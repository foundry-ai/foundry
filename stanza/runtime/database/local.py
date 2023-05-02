from stanza.runtime.database import Database, Table

from pathlib import Path

class LocalDatabase(Database):
    def __init__(self, path):
        self.path = Path(path)

    def open(self, name=None):
        return LocalTable(self.path / name)

    # create a local table
    def create(self, name=None):
        return LocalTable(self.path / name)

class LocalTable(Table):
    def __init__(self, path):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)

    def open(self, name=None):
        return LocalTable(self.path / name)
    
    def add(self, name, value):
        pass