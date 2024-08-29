from functools import partial

from foundry.train.reporting import Reportable, as_log_dict

import foundry.util
import jax
import logging
import IPython.display

logger = logging.getLogger("foundry.train")

def _log_cb(logger_inst, iteration, data_dict, reportable_dict):
    logger_inst = logger_inst or logger
    for k, v in data_dict.items():
        logger_inst.info(f"{iteration: >6} | {k}: {v}")
    for k, r in reportable_dict.items():
        IPython.display.display(r)

@partial(jax.jit,
        static_argnames=("logger", "join", "prefix", "suffix")
    )
def log(iteration, *data, logger=None, join=".", prefix=None, suffix=None):
    data, reportables = as_log_dict(*data, join=join, prefix=prefix, suffix=suffix)
    jax.experimental.io_callback(
        partial(_log_cb, logger), None, 
        iteration, data, reportables,
        ordered=True
    )