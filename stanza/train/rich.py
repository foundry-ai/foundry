from stanza.train import Callback
from stanza.util.dataclasses import dataclass, field, replace

from jax.experimental.host_callback import id_tap, barrier_wait

from rich.progress import (
    Progress, TextColumn, BarColumn,
    ProgressColumn, TimeRemainingColumn
)
from rich.text import Text
from rich.live import Live
from rich.style import Style

_REPORTERS = {}
_counter = 0

@dataclass(jax=True)
class RichCallback(Callback):
    reporter_id: int

    @staticmethod
    def cpu_callback(args, _):
        rid, state = args
        rid = rid.item()
        _REPORTERS[rid].handle(state)

    def __call__(self, state):
        # Don't load the parameter state to the CPU
        state = replace(state, rng_key=None,
            fn_params=None, fn_state=None, opt_state=None)
        id_tap(self.cpu_callback, (self.reporter_id, state))

class RichReporter:
    def __init__(self, iter_interval):
        self.iter_interval = iter_interval

        global _counter
        _counter = _counter + 1
        self.reporter_id = _counter

        self.progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(finished_style=Style(color="green")),
            MofNColumn(),
            TimeRemainingColumn()
        )
        self.epoch_task = None
        self.iter_task = None
        self.live = Live(self.progress)

    def handle(self, state):
        iteration = state.total_iteration.item()
        max_iter = state.max_iteration.item()
        epoch = state.epoch.item()
        max_epoch = state.max_epoch.item() if state.max_epoch else None

        if self.epoch_task is None:
            self.epoch_task = self.progress.add_task("Epoch", total=max_epoch)
        if self.iter_task is None:
            self.iter_task = self.progress.add_task("Iteration", total=max_iter)
        self.progress.update(self.iter_task,
            completed=iteration, total=max_iter)
        self.progress.update(self.epoch_task,
            completed=epoch, total=max_epoch)
        self.live.refresh()
    
    def __enter__(self):
        _REPORTERS[self.reporter_id] = self
        self.live.__enter__()
        return RichCallback(self.iter_interval, _counter)
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        barrier_wait()
        del _REPORTERS[self.reporter_id]
        self.live.__exit__(exc_type, exc_value, exc_traceback)

class MofNColumn(ProgressColumn):
    def __init__(self):
        super().__init__()

    def render(self, task: "Task") -> Text:
        completed = int(task.completed)
        total = int(task.total) if task.total is not None else "?"
        total_width = len(str(total))
        return Text(
            f"{completed:{total_width}d}/{total}",
            style="progress.percentage",
        )