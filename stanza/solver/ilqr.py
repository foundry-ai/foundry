from typing import Callable

from stanza.util.dataclasses import dataclass, field
from stanza.solver import IterativeSolver
from stanza.policy.mpc import MinimizeMPC

@dataclass(jax=True)
class iLQR(IterativeSolver):
    fun: Callable

    def update(self, params, state, ):
        pass