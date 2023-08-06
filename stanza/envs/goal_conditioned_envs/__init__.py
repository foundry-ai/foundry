from stanza.envs import Environment
from stanza.envs.pendulum import PendulumEnv
from typing import Any
from stanza.envs.goal_conditioned_envs.gc_pendulum import make_gc_pendulum_env

def create_gc(env : Environment, gs_sampler : Any, **kwargs):
    if isinstance(env, PendulumEnv):
        return  make_gc_pendulum_env(env, gs_sampler)