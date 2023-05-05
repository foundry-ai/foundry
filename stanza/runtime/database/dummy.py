from stanza.runtime.database import Database

class DummyDatabase(Database):
    def __init__(self, name=None, parent=None):
        self._name = name
        self._parent = parent
        self._children = set()

    @property
    def name(self):
        return self._name

    @property
    def parent(self):
        return self._parent

    @property
    def children(self):
        return self._children
    
    def has(self, name):
        return name in self._children

    def open(self, name):
        self._children.add(name)
        return DummyDatabase(name, self)

    def add(self, name, value):
        self._children.add(name)