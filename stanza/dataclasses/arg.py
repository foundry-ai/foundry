from stanza.dataclasses import dataclass

class ArgField:
    def __init__(self, name, arg_path):
        self.name = name
        self.arg_path = arg_path
    
    def populate(self, parser):
        pass

    def unpack(args):
        pass

    @staticmethod
    def from_type(type, *args, **kwargs):
        if dataclass.is_dataclass(type):
            pass

class DataclassArgField(ArgField):
    def __init__(self, dc, name, arg_path):
        pass

class ArgParser:
    def __init__(self):
        self.fields = []

    def add(self, name, type):
        self.fields.append(ArgField(name))