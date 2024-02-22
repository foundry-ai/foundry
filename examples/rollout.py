import logging
logger = logging.getLogger("examples.rollout")
logger.setLevel(logging.DEBUG)

import stanza.policy as policy
import jax.numpy as jnp

def system(x, u, _rng_key):
    return x + u

def run():
    actions = policy.Actions(jnp.array([1, 2, 3]))
    out = actions(policy.PolicyInput(0))
    logger.info(f"action: {out.action}")
    trajectory = policy.rollout(system, 10, actions)
    logger.info(f"trajectory: {trajectory}")

if __name__ == "__main__":
    from stanza.runtime import setup_logger
    setup_logger()
    run()