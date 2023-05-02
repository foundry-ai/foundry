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
video = jax.vmap(env.render)(rollout.states)
import ffmpegio
ffmpegio.video.write('video.mp4', 28, video, 
    overwrite=True, loglevel='quiet')

# dataset = pusht.expert_dataset()
policy = pusht.pretrained_policy()
t = time.time()
rollout = policies.rollout(env.step, x0, policy,
            length=250*10,
            policy_rng_key=PRNGKey(42), last_state=False)
logger.info('took: {:02}s to rollout', time.time() - t)

# render every 10th state
vis_states = jax.tree_util.tree_map(lambda x: x[::10], rollout.states)
video = jax.vmap(env.render)(vis_states)
import ffmpegio
ffmpegio.video.write('trained_video.mp4', 28, video, 
    overwrite=True, loglevel='quiet')

# rollout = policies.rollout(env.step, x0, policy, 
#                            policy_rng_key=PRNGKey(42),
#                             length=2)
# logger.info('rollout: {}', rollout)
# data = dataset[:1000].data
# video = jax.vmap(env.render)(data[0])
# ffmpegio.video.write('dataset_video.mp4', 60, video, 
#     overwrite=True, loglevel='quiet')