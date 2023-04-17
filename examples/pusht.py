import stanza.envs as envs
import stanza.policies as policies
from stanza.util.logging import logger

import jax.numpy as jnp
import jax
from jax.random import PRNGKey
from jax.tree_util import tree_map

pusht = envs.create('pusht')
x0 = pusht.reset(PRNGKey(0))
x0_img = pusht.render(x0)

logger.info("x0: {}", x0)

input = pusht.sample_action(PRNGKey(1))
inputs = tree_map(lambda x: jnp.repeat(x[jnp.newaxis,...],10,0), input)

rollout = policies.rollout_inputs(pusht.step, x0, inputs, last_state=False)
logger.info('rollout: {}', rollout)

from stanza.envs.pusht import expert_dataset
dataset = expert_dataset()

logger.info("{} {}", jax.tree_util.tree_map(lambda x: x.shape, rollout.states),
    rollout.actions.shape)
video = jax.vmap(pusht.render)(rollout.states, rollout.actions)
import ffmpegio
ffmpegio.video.write('video.mp4', 4, video, 
    overwrite=True, loglevel='quiet')

# make a video of the first 100 states in the dataset
data = dataset[:1000].data
video = jax.vmap(pusht.render)(data[0],data[1])
ffmpegio.video.write('dataset_video.mp4', 60, video, 
    overwrite=True, loglevel='quiet')