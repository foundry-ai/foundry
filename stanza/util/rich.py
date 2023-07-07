from stanza.dataclasses import dataclass
from rich.live import Live as RichLive
from rich.text import Text as RichText
from rich.console import Group as RichGroup
from rich.progress import (
    Progress as RichProgress,
    ProgressColumn as ProgressColumn,
    TextColumn, TimeRemainingColumn,
    TimeElapsedColumn, BarColumn
)
from rich.table import Table as RichTable
from rich.style import Style

from stanza import partial, Partial
from typing import Dict, Any

import jax
import abc
import jax.numpy as jnp

from abc import abstractmethod
from jax.experimental.host_callback import id_tap, barrier_wait

_DISPLAYS = {}
_counter = 0

def _cpu_callback(name, interval, args, _):
    id, extracted = args
    id = id.item()
    display = _DISPLAYS[id]
    display.update(name, interval, extracted)

def _split(iterable):
    return tuple(zip(*iterable))

@dataclass(jax=True)
class RichHook:
    id: jnp.array
    extractors: Dict[str, Any]

    def hook(self, name, hook_state, state):
        intervals = self.extractors[name]
        hook_state = {} if hook_state is None else hook_state
        new_hook_state = dict()
        for (interval, extractors) in intervals.items():
            hs = hook_state.get(interval, [None]*len(extractors))
            hs, values = _split([v(hs, state) for hs, v in zip(hs, extractors)])
            def cb(state):
                state = id_tap(partial(_cpu_callback, name, interval), 
                            (self.id, values), result=state)
                return state
            do_cb = jnp.logical_or(state.iteration % interval == 0,
                                state.iteration == state.max_iterations)
            state = jax.lax.cond(do_cb, cb, lambda x: x, state)
            new_hook_state[interval] = hs
        return None, state
    
    def __getattr__(self, name: str) -> Any:
        if name in ["id", "extractors"]:
            super().__getattribute__(name)
        if not name in self.extractors:
            raise AttributeError(f"hook {name} not defined!")
        hook = partial(Partial(type(self).hook, self), name)
        return hook
    
class Element(abc.ABC):
    @abstractmethod
    def update(self, updates):
        ...

    @abstractmethod
    def extract(self, element_state, *args):
        ...

class MofNColumn(ProgressColumn):
    def __init__(self):
        super().__init__()

    def render(self, task) -> RichText:
        completed = int(task.completed)
        total = int(task.total) if task.total is not None else "?"
        total_width = len(str(total))
        return RichText(
            f"{completed:{total_width}d}/{total}",
            style="progress.percentage",
        )

class Progress(Element):
    def __init__(self, task_name="", columns=None):
        if columns is None:
            columns = [
                TextColumn("[progress.description]{task.description}"),
                BarColumn(finished_style=Style(color="green")),
                MofNColumn(),
                TimeRemainingColumn(),
                TimeElapsedColumn()
            ]
        self.task_name = task_name
        self.widget = RichProgress(*columns)
        self.task = None

class LoopProgress(Progress):
    def __init__(self, task_name="Iteration", columns=None):
        super().__init__(task_name, columns)

    def update(self, updates):
        completed, total = updates
        if self.task is None:
            self.task = self.widget.add_task(self.task_name,
                completed=completed, total=total)
        self.widget.update(self.task,
            completed=completed,
            total=total)

    def extract(self, element_state, state):
        return element_state, (state.iteration, state.max_iterations)

class EpochProgress(Progress):
    def __init__(self, task_name="Epoch", columns=None):
        super().__init__(task_name, columns)

    def update(self, updates):
        completed, total = updates
        if self.task is None:
            self.task = self.widget.add_task(self.task_name,
                completed=completed, total=total)
        self.widget.update(self.task,
            completed=completed,
            total=total)

    def extract(self, element_state, state):
        return element_state, (state.epoch, state.max_epochs)
    

def _flatten(d):
    for (k, v) in d.items():
        if isinstance(v, dict):
            for (k2, v2) in _flatten(v):
                yield (f"{k}.{k2}", v2)
        else:
            yield (k, v)

class StatisticsTable(Element):
    def __init__(self, stats=None):
        super().__init__()
        self.widget = RichTable()
        self.stats = stats
    
    def update(self, stats):
        self.widget.rows = []
        self.widget.columns = []
        self.widget.add_column("Statistic")
        self.widget.add_column("Value")

        for (stat, val) in stats.items():
            cells = [RichText(f"{val:5f}")]
            self.widget.add_row(stat, *cells)

    def extract(self, element_state, state):
        if state.last_stats is None:
            return element_state, {}
        stats = dict(_flatten(state.last_stats))
        if self.stats is not None:
            stats = {k: v for (k, v) in stats.items() if k in self.stats}
        # take the mean of the stats
        stats = jax.tree_map(lambda x: jnp.mean(x), stats)
        return element_state, stats

class ConsoleDisplay:
    def __init__(self):
        self.intervals = {}
        self.groups = {}
        self.elements = []
        self.live = None

        global _counter
        _counter = _counter + 1
        self.display_id = _counter

    def add(self, group_name, element, interval=1):
        if self.display_id in _DISPLAYS:
            raise ValueError("Cannot not add display elements while active")
        intervals = self.groups.setdefault(group_name, {})
        elems = intervals.setdefault(interval, [])
        elems.append(element)
        self.elements.append(element)

    def update(self, group, interval, values):
        intervals = self.groups[group]
        elems = intervals[interval]
        for (e,v) in zip(elems, values):
            e.update(v)
        self.live.refresh()

    def __enter__(self):
        if self.live is not None:
            raise ValueError("Cannot make display active recursively")
        _DISPLAYS[self.display_id] = self
        from stanza.util.logging import console

        self.live = RichLive(
            RichGroup(*[r.widget for r in self.elements]),
            console=console)
        self.live.__enter__()
        extractors = {
            group_name: {
                i : [v.extract for v in l] \
                    for (i,l) in group.items()
            } for (group_name, group) in self.groups.items()}
        return RichHook(self.display_id, extractors)

    def __exit__(self, exc_type, exc_value, exc_traceback):
        barrier_wait()
        del _DISPLAYS[self.display_id]
        self.live.__exit__(exc_type,
            exc_value, exc_traceback)
        self.live = None