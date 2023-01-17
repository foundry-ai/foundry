from jinx.experiment import Repo, Experiment, Run, remap, \
                            Video, Figure

class DummyRepo(Repo):
    def __init__(self):
        pass

    def experiment(self, name):
        return DummyExperiment()
    

class DummyExperiment(Experiment):
    def create_run(self, name=None):
        return DummyRun()

class DummyRun(Run):
    def sub_run(self, prefix):
        return self
    
    def _log(self, data):
        pass