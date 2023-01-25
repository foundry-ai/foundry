import jax
from termcolor import colored

from functools import partial

import jax.experimental.host_callback

# Topic management




# ---------------- Logging ---------------

TRACE = 'TRACE'
DEBUG = 'DEBUG'
INFO =  'INFO'
WARN =  'WARN'
ERROR = 'ERROR'

LEVEL_COLORS = {
    TRACE: 'green',
    DEBUG: 'cyan',
    INFO: 'white',
    WARN: 'yellow',
    ERROR: 'red'
}

# A jax-compatible logger
# This will bypass logging at compile time
# if the selected "topic" have not been enabled
class JaxLogger:
    def __init__(self):
        self._initialized = False
        self._allowed = []

    def init(self):
        self._initialized = True

    def _allow(self, level, topic):
        return True

    # Host-side logging
    def _log(self, level, topic, msg, jax_args, tracing=False):
        args, kwargs = jax_args
        msg = msg.format(*args, **kwargs)

        topic = colored(f'{topic:8}', 'blue')
        level = colored(f'{level:6}', LEVEL_COLORS.get(level, 'white'))
        msg = colored(msg, 'white')
        if tracing:
            msg = colored('<Tracing> ', 'yellow') + msg
        print(f' {topic} | {level} - {msg}')

    def log(self, level, *args, **kwargs):
        if len(args) == 0:
            topic = kwargs.get('topic', '')
            msg = kwargs.get('msg', '')

        # allow for logging with just message + args (without topic)
        if len(args) == 1 or not isinstance(args[1], str):
            msg = args[0]
            topic = kwargs.get('topic', '')
            args = args[1:]
        else:
            topic, msg = args[:2]
            args = args[2:]

        if not self._initialized:
            self._log(WARN, 'logger', 
                'Logger not yet initialized, using default initialization',
                ([], {}))
            self.init()
        # Short-circuit at compile time!
        if self._allow(level, topic):
            # if isinstance(jax.numpy.array(0), jax.core.Tracer):
            #    self._log(level, topic, msg, (args, kwargs), tracing=True)
            self._log(level, topic, msg, (args, kwargs), tracing=False)
            # jax.debug.callback(
            #             partial(self._log, level, topic, msg), 
            #             (args, kwargs), ordered=True)

    def trace(self, *args, **kwargs):
        return self.log(TRACE, *args, **kwargs)

    def debug(self, *args, **kwargs):
        return self.log(DEBUG, *args, **kwargs)

    def info(self, *args, **kwargs):
        return self.log(INFO, *args, **kwargs)

    def warn(self, *args, **kwargs):
        return self.log(WARN, *args, **kwargs)

    def error(self, *args, **kwargs):
        return self.log(ERROR, *args, **kwargs)

# The default logger (note that it is not initialized!)
logger = JaxLogger()

# ---------------- ProgressBars ---------------
import tqdm

class ProgressBar:
    def __init__(self, topic, total):
        self.topic = topic
        #jax.debug.callback(self._create, total, ordered=True)

    def inc(self, amt=1, stats={}):
        # TODO: We need to get the debug callback's vmap-transform
        # unrolling
        #jax.debug.callback(self._inc, amt, stats, ordered=True)
        pass

    # TODO: Figure out what to do about close()
    # getting optimized out...
    def close(self):
        pass
        #jax.debug.callback(self._close, ordered=True)

    def _create(self, total):
        self.bar = tqdm.tqdm(total=total)
        s = ''
        topic = colored(f'{self.topic:8}', 'blue')
        self.bar.set_description(f' {topic} | {s:6} ')

    def _inc(self, amt, stats):
        postfix = ', '.join([f'{k}: {v.item():8.3}' for (k,v) in stats.items()])
        self.bar.set_postfix_str(postfix)
        self.bar.update(amt)

    def _close(self):
        self.bar.close()

    def __enter__(self):
        return self
    
    def __exit__(self, _0, _1, _2):
        self.close()

# Handy ergonomic function
def pbar(topic, total):
    return ProgressBar(topic, total)