from typing import Callable

from stanza.util.dataclasses import dataclass
from stanza.solver import IterativeSolver

@dataclass(jax=True)
class iLQR(IterativeSolver):
    fun: Callable

    def update(self, params, state, ):
        pass