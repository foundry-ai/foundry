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

rollout = policies.rollout_inputs(pusht.step, x0, inputs)
logger.info('rollout: {}', rollout)

from stanza.envs.pusht import expert_dataset
dataset = expert_dataset()

# encode video of the pusht
video = jax.vmap(pusht.render)(rollout.states)
import ffmpegio
ffmpegio.video.write('video.mp4', 4, video, 
    overwrite=True, loglevel='quiet')

# make a video of the first 100 states in the dataset
state_data = dataset[:1000].data[0]
video = jax.vmap(pusht.render)(state_data)
ffmpegio.video.write('dataset_video.mp4', 60, video, 
    overwrite=True, loglevel='quiet')