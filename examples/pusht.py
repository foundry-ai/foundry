import stanza.envs as envs
import stanza.policies as policies
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
policy = policies.chain_transforms(
    policies.SampleRateTransform(50),
    # PositionObs *must* come before
    # PositionControl since PositionControl
    # relies on having the full state as observation
    # pusht.PositionObsTransform(),
    pusht.PositionControlTransform()
)(policies.Actions(target_pos))
rollout = policies.rollout(env.step, x0, policy, last_state=False)

# Load in the data!
data = pusht.expert_data()
from stanza.data.trajectory import chunk_trajectory
from stanza.data import PyTreeData
from functools import partial
# flat = data.flatten()
# print(len(flat))

chunked = data.map(
    partial(chunk_trajectory, 
    obs_chunk_size=2, action_chunk_size=8))
data = PyTreeData.from_data(chunked.flatten(), chunk_size=2048)
print(jax.tree_util.tree_map(lambda x: x.shape, data.data))
print(f"Got data {len(data)}")
# chunked = PyTreeData.from_data(chunked, chunk_size=2048)