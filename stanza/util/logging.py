import jax
import stanza

from functools import partial

import jax.experimental.host_callback
import rich.console
from rich.markup import escape

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

console = rich.console.Console()

JAX_PLACEHOLDER = object()

# A jax-compatible logger
# This will bypass logging at compile time
# if the selected "topic" have not been enabled
class JaxLogger:
    def __init__(self):
        pass

    # Host-side logging
    def _log_callback(self, level, msg, reg_comp, jax_comp,
                        tracing=False, highlight=True, stack_offset=3):
        reg_args, reg_kwargs = reg_comp
        jax_args, jax_kwargs = jax_comp

        # reassemble args, kwargs
        args = []
        jax_iter = iter(jax_args)
        for a in reg_args:
            if a is JAX_PLACEHOLDER:
                args.append(next(jax_iter))
            else:
                args.append(a)
        kwargs = dict(reg_kwargs)
        kwargs.update(jax_kwargs)

        msg = msg.format(*args, **kwargs)
        level_color = LEVEL_COLORS.get(level, 'white')
        if tracing:
            msg = '[yellow]<Tracing>[/yellow] ' + msg

        # a version of console.log() which handles the stack frame correctly
        console.log(f'[{level_color}]{level:6}[/{level_color}] - {msg}', 
            highlight=highlight, _stack_offset=stack_offset)

    def log(self, level, msg, *args, highlight=True, show_tracing=False, **kwargs):
        # split the arguments and kwargs
        # based on whether they are jax-compatible types or not
        reg_args = []
        jax_args = []
        for a in args:
            if stanza.is_jaxtype(type(a)):
                jax_args.append(a)
                reg_args.append(JAX_PLACEHOLDER)
            else:
                reg_args.append(a)
        reg_kwargs = {}
        jax_kwargs = {}
        for (k,v) in kwargs.items():
            if stanza.is_jaxtype(type(v)):
                jax_kwargs[k] = v
            else:
                reg_kwargs[k] = v
        tracing = isinstance(jax.numpy.array(0), jax.core.Tracer)
        if tracing and show_tracing:
            self._log_callback(level, msg, (reg_args, reg_kwargs), (jax_args, jax_kwargs), 
                            tracing=True, stack_offset=3)

        jax.debug.callback(partial(self._log_callback, level, msg,
                                   (reg_args, reg_kwargs),
                                   highlight=highlight, stack_offset=10),  
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
        self.total = total
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
        pass

    def _inc(self, amt, stats):
        postfix = ', '.join([f'{k}: {v.item():8.3}' for (k,v) in stats.items()])
        console.log(f'{self.topic} - {postfix}')

    def _close(self):
        pass

    def __enter__(self):
        return self
    
    def __exit__(self, _0, _1, _2):
        self.close()

# Handy ergonomic function
def pbar(topic, total):
    return ProgressBar(topic, total)