import stanza.envs as envs
import stanza.policy as policy
import stanza.envs.pusht as pusht
from stanza.util.logging import logger

import time
import jax.numpy as jnp
import jax
import sys
from jax.random import PRNGKey
from jax.tree_util import tree_map

env = envs.create('pusht')
x0 = env.reset(PRNGKey(13))
x0_img = env.render(x0)

logger.info("x0: {}", x0)

target_pos = jnp.array([
    [150, 150],
    [400, 150],
    [400, 400],
    [150, 400],
    [150, 150],
])
# Create a policy which uses
# the low-level feedback gains
# and runs the target_pos at 20 hz
policy = policy.chain_transforms(
    policy.SampleRateTransform(50),
    # PositionObs *must* come before
    # PositionControl since PositionControl
    # relies on having the full state as observation
    pusht.PositionObsTransform(),
    pusht.PositionControlTransform()
)(policy.Actions(target_pos))
rollout = policy.rollout(env.step, x0, policy, last_state=False)