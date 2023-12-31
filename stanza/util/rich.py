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

from typing import Dict, List, Any

import abc
from abc import abstractmethod
from weakref import WeakValueDictionary

from stanza.util.loop import Hook
from stanza.dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import jax.experimental

# The live management, so
# we can have multiple live environments

_LIVE = None
_LIVE_GROUP = None
_LIVE_STACK = []

def push_live(*renderables):
    _LIVE_STACK.append(renderables)
    r = [r for s in _LIVE_STACK for r in s]
    global _LIVE
    global _LIVE_GROUP
    if _LIVE is None:
        _LIVE_GROUP = RichGroup(*r)
        _LIVE = RichLive(_LIVE_GROUP) 
        # enter the live environment
        _LIVE.__enter__()
    else:
        _LIVE_GROUP._renderables = r

def pop_live():
    _LIVE_STACK.pop()
    global _LIVE
    global _LIVE_GROUP
    if len(_LIVE_STACK) == 0:
        _LIVE.__exit__(None, None, None)
        _LIVE = None
        _LIVE_GROUP = None
    else:
        r = [r for s in _LIVE_STACK for r in s]
        _LIVE_GROUP._renderables = r


# The host-side object
@dataclass
class _ConsoleDisplay:
    elements: List = field(default_factory=list)
    intervals: List = field(default_factory=dict)

    def add_element(self, element, interval):
        self.elements.append(element)
        self.intervals.setdefault(interval, []).append(element)

    # Make "hashable" using the id
    def __hash__(self):
        return id(self)

@dataclass(jax=True)
class ConsoleDisplay(Hook):
    display: _ConsoleDisplay = field(
        default_factory=_ConsoleDisplay,
        jax_static=True
    )

    def add(self, elements, interval=1):
        self.display.add_element(elements, interval)
    
    @staticmethod
    def _host_make_live(disp):
        r = [r for e in disp.display.elements 
               for r in e.renderables]
        push_live(*r)

    @staticmethod
    def _host_stop_live():
        pop_live()
    
    @staticmethod
    def _host_update(disp, interval, updates):
        elements = disp.display.intervals[interval.item()]
        for (element, update) in zip(elements, updates):
            element.handle_updates(update)

    def init(self, state):
        jax.experimental.io_callback(self._host_make_live, (), 
                                     self, ordered=True)
        element_states = []
        for element in self.display.elements:
            element_states.append(element.init_state(state))
        hook_state = -1, element_states
        return hook_state, state
    
    def run(self, hook_state, state):
        iteration, element_states = hook_state
        def do_update():
            new_es = []
            elem_states = {}
            for (element, es) in zip(self.display.elements, element_states):
                new_es.append(
                    element.update_state(es, state)
                )
                elem_states[id(element)] = new_es[-1]
            for (interval, elements) in self.display.intervals.items():
                def do_callback():
                    es = [elem_states[id(e)] for e in elements]
                    updates = list([e.compute_updates(es, state) for e, es in zip(elements, es)])
                    jax.experimental.io_callback(
                        self._host_update, (), self, interval, updates, ordered=True
                    )
                jax.lax.cond(iteration % interval == 0,
                    do_callback, lambda: None)
            return state.iteration, new_es
        new_hook_state = jax.lax.cond(iteration != state.iteration,
            do_update, lambda: hook_state)
        return new_hook_state, state

    def finalize(self, hook_state, state):
        _, element_states = hook_state
        # do a final update before closing the live environment
        elem_states = {}
        for (e, es) in zip(self.display.elements, element_states):
            elem_states[id(e)] = es

        for (interval, elements) in self.display.intervals.items():
            es = [elem_states[id(e)] for e in elements]
            updates = list([e.compute_updates(es, state) for e, es in zip(elements, es)])
            jax.experimental.io_callback(
                self._host_update, (), self, interval, updates, ordered=True
            )

        jax.experimental.io_callback(self._host_stop_live, (), ordered=True)
        return hook_state, state

class Element(abc.ABC):
    @property
    def renderables(self) -> List:
        raise NotImplementedError()

    @abstractmethod
    def compute_updates(self, element_state, state):
        ...

    @abstractmethod
    def handle_updates(self, updates):
        ...

    def init_state(self, state):
        return None
    
    def update_state(self, element_state, state):
        return element_state

    def finalize_state(self, element_state):
        return element_state
    

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

class ProgressBar(Element):
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
    
    @property
    def renderables(self):
        return [self.widget]
    
    def handle_updates(self, updates):
        completed, total = updates
        if self.task is None:
            self.task = self.widget.add_task(self.task_name,
                completed=completed, total=total)
        self.widget.update(self.task,
            completed=completed,
            total=total)
        return super().handle_updates(updates)

class LoopProgressBar(ProgressBar):
    def __init__(self, task_name="Iteration", columns=None):
        super().__init__(task_name, columns)

    def compute_updates(self, element_state, state):
        return (state.iteration, state.max_iterations)

class EpochProgressBar(ProgressBar):
    def __init__(self, task_name="Epoch", columns=None):
        super().__init__(task_name, columns)

    def compute_updates(self, element_state, state):
        return (state.epoch, state.max_epochs)
    
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
    
    @property
    def renderables(self):
        return [self.widget]
    
    def handle_updates(self, stats):
        self.widget.rows = []
        self.widget.columns = []
        self.widget.add_column("Statistic")
        self.widget.add_column("Value")

        for (stat, val) in stats.items():
            cells = [RichText(f"{val:5f}")]
            self.widget.add_row(stat, *cells)

    def compute_updates(self, element_state, state):
        if state.last_stats is None:
            return {}
        stats = dict(_flatten(state.last_stats))
        if self.stats is not None:
            stats = {k: v for (k, v) in stats.items() if k in self.stats}
        # take the mean of the stats
        stats = jax.tree_map(lambda x: jnp.mean(x), stats)
        return stats


