import stanza.envs as envs
import stanza.policies as policies
import stanza.envs.pusht as pusht
from stanza.util.logging import logger

import jax.numpy as jnp
import jax
from jax.random import PRNGKey
from jax.tree_util import tree_map

env = envs.create('pusht')
x0 = env.reset(PRNGKey(0))
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
policy, ll_policy = pusht.pretrained_policy()

obs = pusht.PushTPositionObs(
    jnp.array([235., 124.]),
    jnp.array([335., 128.]),
    jnp.array(5.37825788)
)
obs = jax.tree_util.tree_map(lambda a,b: jnp.stack((a,b), axis=0), obs, obs)
output = ll_policy(policies.PolicyInput(obs, None, PRNGKey(42)))

output = ll_policy(policies.PolicyInput(obs, None, PRNGKey(42)))
jax.tree_util.tree_map(
    lambda x: x.block_until_ready(),
    output
)
logger.info("{}", output)

# rollout = policies.rollout(env.step, x0, policy,
#                     length=100*20, policy_rng_key=PRNGKey(42), last_state=False)
# video = jax.vmap(env.render)(rollout.states)
# import ffmpegio
# ffmpegio.video.write('pretrained_video.mp4', 28, video, 
#     overwrite=True, loglevel='quiet')

# rollout = policies.rollout(env.step, x0, policy, 
#                            policy_rng_key=PRNGKey(42),
#                             length=2)
# logger.info('rollout: {}', rollout)
# data = dataset[:1000].data
# video = jax.vmap(env.render)(data[0])
# ffmpegio.video.write('dataset_video.mp4', 60, video, 
#     overwrite=True, loglevel='quiet')