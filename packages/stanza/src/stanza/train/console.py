from stanza.train.reporting import as_log_dict
from functools import partial

import jax

import logging
logger = logging.getLogger("stanza.train")

def _log_cb(logger_inst, iteration, data_dict):
    logger_inst = logger_inst or logger
    for k, v in data_dict.items():
        logger_inst.info(f"{iteration: >6} | {k}: {v}")

@partial(jax.jit,
        static_argnames=("logger", "join", "prefix", "suffix")
    )
def log(iteration, *data, logger=None, join=".", prefix=None, suffix=None):
    data, reportables = as_log_dict(*data, join=join, prefix=prefix, suffix=suffix)
    jax.experimental.io_callback(
        partial(_log_cb, logger), None, 
        iteration, data, ordered=True
    )