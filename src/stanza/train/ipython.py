from stanza.util.ipython import display
from stanza.train.reporting import Reportable, dict_flatten

def display_logger(*hooks, 
        metrics=False, step_info=False,
        prefix=None, suffix=None, graph_area=None):
    def logger(rng, state, log=None, **kwargs):
        r = []
        if log is not None: r.append(log)
        if metrics: r.append(state.metrics)
        if step_info: r.append({
            "iteration": state.iteration,
            "epoch": state.epoch,
            "epoch_iteration": state.epoch_iteration
        })
        for hook in hooks: r.append(hook(rng, state, **kwargs))
        flattened = dict_flatten(
            *r,
            prefix=prefix, suffix=suffix
        )
        for k,v in flattened.items():
            if isinstance(v, Reportable):
                display(v)
    return logger