from stanza.runtime.database import Database, Table

class DummyDatabase(Database):
    def __init__(self):
        pass

    def open(self, name=None):
        return DummyTable()

class DummyTable(Table):
    def open(self, name=None):
        return DummyTable()