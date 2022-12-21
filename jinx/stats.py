from loguru import logger

from jax.experimental.host_callback import id_tap

class Report:
    def __init__(self, func, arg):
        self.func = func
        self.arg = arg
    
    def attach(self, val):
        return id_tap(lambda a,_: self.func(a), self.arg, result=val)

class Reporter:
    def __init__(self):
        pass
    
    def send(self, event):
        pass


class LoggerReporter(Reporter):
    def __init__(self, prefix=''):
        self.prefix = prefix

    def print(self, e):
        fmt = ', '.join([f'{k}: {v}' for k,v in e.items()])
        logger.info(f'{self.prefix} {fmt}')
    
    def send(self, event):
        return Report(self.print, event)