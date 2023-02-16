from stanza.runtime.database import Database, Experiment, Run, remap, \
                                    Video, Figure

class DummyDatabase(Database):
    def __init__(self):
        pass

    def experiment(self, name):
        return DummyExperiment()
    

class DummyExperiment(Experiment):
    def create_run(self, name=None):
        return DummyRun()

class DummyRun(Run):
    def __init__(self):
        pass
        # self.config = AttrDict()
        # self.summary = AttrDict()

    def sub_run(self, prefix):
        return self
    
    def _log(self, data):
        pass