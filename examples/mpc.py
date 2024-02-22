from stanza.envs import env_registry
from stanza.util.random import PRNGSequence

from stanza.policy.mpc import MPC
from stanza.policy import PolicyInput

from jax.random import PRNGKey

import logging
logger = logging.getLogger("examples.mpc")
logger.setLevel(logging.DEBUG)

def run():
    rng = PRNGSequence(42)
    env = env_registry.create("quadrotor_2d")
    x0 = env.reset(next(rng))

    mpc = MPC(
        action_sample=env.sample_action(PRNGKey(0)),
        cost_fn=env.cost,
        model_fn=env.step,
        horizon_length=20
    )
    mpc_action = mpc(PolicyInput(x0))
    logger.info(f"Action: {mpc_action}")

if __name__ == "__main__":
    from stanza.runtime import setup
    setup()
    run()