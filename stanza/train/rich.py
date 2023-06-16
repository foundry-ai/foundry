from stanza.dataclasses import dataclass, field, replace

import functools
import jax
import jax.numpy as jnp

from jax.experimental.host_callback import id_tap, barrier_wait

from rich.progress import (
    Progress, TextColumn, BarColumn,
    ProgressColumn, TimeRemainingColumn,
    TimeElapsedColumn
)
from rich.console import Group
from rich.text import Text
from rich.table import Table
from rich.live import Live
from rich.style import Style

_REPORTERS = {}
_counter = 0

@dataclass(jax=True)
class RichCallback:
    iter_interval: int
    average_window: int
    reporter_id: int

    @staticmethod
    def cpu_state_callback(args, _):
        rid, state = args
        rid = rid.item()
        _REPORTERS[rid]._handle_state(state)

    @staticmethod
    def cpu_stat_callback(args, _, split):
        rid, stats = args
        rid = rid.item()
        _REPORTERS[rid].update_stats(stats, split)
    
    def _do_state_callback(self, state):
        cpu_state = replace(state, rng_key=None,
            fn_params=None, fn_state=None, opt_state=None)
        return id_tap(self.cpu_state_callback, (self.reporter_id, cpu_state), result=state)
    
    def update_stats(self, stats, split="Train"):
        cb = functools.partial(self.cpu_stat_callback, split=split)
        id_tap(cb, (self.reporter_id, stats))

    def __call__(self, hs, state):
        # Don't load the parameter state to the CPU
        if hs is not None and state.last_stats is not None:
            hs = jax.tree_util.tree_map(
                lambda a, b: 1/self.average_window * b + (1 - 1/self.average_window)*a,
                hs, state.last_stats
            )
        if hs is None and state.last_stats is not None:
            hs = state.last_stats

        new_state = jax.lax.cond(
            jnp.logical_or(
                jnp.logical_and(state.total_iteration % self.iter_interval == 0,
                            state.last_stats is not None),
                # When we have reached the end, do a callback
                # so that we can finish the progress bars
                state.epoch == state.max_epoch
            ),
            self._do_state_callback,
            lambda x: x, replace(state, last_stats=hs)
        )
        return hs, replace(new_state, last_stats=state.last_stats)

class RichReporter:
    def __init__(self, iter_interval=20, average_window=20):
        self.iter_interval = iter_interval
        self.average_window = average_window

        global _counter
        _counter = _counter + 1
        self.reporter_id = _counter

        self.initialized = False

        self.epoch_task = None
        self.iter_task = None
        self.progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(finished_style=Style(color="green")),
            MofNColumn(),
            TimeRemainingColumn(),
            TimeElapsedColumn()
        )
        self.table = Table()
        self.live = Live(Group(self.table, self.progress))
        self.split_stats = {}
    
    def update_stats(self, stats, split="Train"):
        self.split_stats[split] = stats
    
    def _update_table(self):
        if not self.split_stats:
            return
        self.table.rows = []
        self.table.columns = []
        self.table.add_column("Statistic")

        stats = {}
        for (i,(split, ss)) in enumerate(self.split_stats.items()):
            self.table.add_column(split)
            for (k,v) in ss.items():
                stats.setdefault(k,{})[i] = v.item()
        for (stat, ss) in stats.items():
            cells = [Text(f"{ss[i]:5f}") if i in ss else None \
                     for i in range(len(self.split_stats))]
            self.table.add_row(stat, *cells)

    def _handle_state(self, state):
        iteration = state.total_iteration.item()
        max_iter = state.max_iteration.item()
        epoch = state.epoch.item()
        max_epoch = state.max_epoch.item() if state.max_epoch else None
        # clear the table
        if state.last_stats is not None:
            self.split_stats['Train'] = state.last_stats
        self._update_table()
        if not self.initialized:
            self.initialized = True
            self.iter_task = self.progress.add_task("Iteration",
                                completed=iteration, total=max_iter)
            self.epoch_task = self.progress.add_task("Epoch",
                                completed=epoch, total=max_epoch)
            # set up the table
        else:
            self.progress.update(self.iter_task,
                completed=iteration, total=max_iter)
            self.progress.update(self.epoch_task,
                completed=epoch, total=max_epoch)
        self.live.refresh()
    
    def __enter__(self):
        _REPORTERS[self.reporter_id] = self
        self.live.__enter__()
        return RichCallback(self.iter_interval, self.average_window, 
                            self.reporter_id)
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        barrier_wait()
        del _REPORTERS[self.reporter_id]
        self.live.__exit__(exc_type, exc_value, exc_traceback)

class MofNColumn(ProgressColumn):
    def __init__(self):
        super().__init__()

    def render(self, task) -> Text:
        completed = int(task.completed)
        total = int(task.total) if task.total is not None else "?"
        total_width = len(str(total))
        return Text(
            f"{completed:{total_width}d}/{total}",
            style="progress.percentage",
        )