import jax

from functools import partial

import jax.experimental.host_callback
import rich

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
        pass

    # Host-side logging
    def _log(self, level, topic, msg, jax_args, tracing=False):
        args, kwargs = jax_args
        msg = msg.format(*args, **kwargs)

        level_color = LEVEL_COLORS.get(level, 'white')
        if tracing:
            msg = '[yellow]<Tracing>[/yellow] ' + msg
        rich.print(f'----| [{level_color}]{level:6}[/{level_color}] - {msg}')

    def log(self, level, msg, *args, **kwargs):
        if len(args) > 0 and isinstance(args[0], str):
            topic = msg
            msg = args[0]
        else:
            topic = ''

        # if we are tracing, still do the print, but
        # note that the function is being traced
        if isinstance(jax.numpy.array(0), jax.core.Tracer):
            self._log(level, topic, msg, (args, kwargs), tracing=True)
        jax.debug.callback(
                    partial(self._log, level, topic, msg), 
                    (args, kwargs),
        ordered=True)

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

class ProgressBar:
    def __init__(self, topic, total):
        self.topic = topic
        jax.debug.callback(self._create, total, ordered=True)

    def inc(self, amt=1, stats={}):
        # TODO: We need to get the debug callback's vmap-transform
        # unrolling
        jax.debug.callback(self._inc, amt, stats, ordered=True)

    # TODO: Figure out what to do about close()
    # getting optimized out...
    def close(self):
        jax.debug.callback(self._close, ordered=True)

    def _create(self, total):
        self.bar = tqdm.tqdm(total=total)
        s = ''
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