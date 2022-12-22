from loguru import logger

from jax.experimental.host_callback import id_tap

def topic_join(a, b):
    if a is None:
        a = ()
    elif not isinstance(a, tuple):
        a = (a,)

    if b is None:
        b = ()
    elif not isinstance(b, tuple):
        b = (b,)

    return a + b

class Reporter:
    def __init__(self):
        self._listeners = []
        self._mapped = {}
    
    def sub(self, handler, topic=None):
        if topic is None:
            l = list(self._listeners)
            l.append(handler)
            self._listeners = l
        else:
            l = list(self._mapped.get(topic, []))
            l.append(handler)
            self._mapped[topic] = l
    
    def attach(self, sub_reporter, topic=None, sub_topic=None):
        sub_reporter.sub(
            lambda t, e: self.send(topic_join(topic, t), e),
            sub_topic
        )

    def send(self, topic, event):
        for l in self._listeners:
            l(topic, event)
        for l in self._mapped.get(topic,[]):
            l(topic, event)
    
    def tap(self, val, topic, event):
        return id_tap(lambda a,_: self.send(topic, a), event, result=val)

class Collector:
    def __init__(self):
        self._events = {}
    
    def appender(self, reporter, topic):
        def append(_, e):
            events = self._events.setdefault(topic, [])
            events.append(e)
        reporter.sub(append, topic)
    
    def updater(self, reporter, topic):
        def update(_, e):
            self._events[topic] = e
        reporter.sub(update, topic)

    def get(self, topic, default=None):
        return self._events.get(topic, default)