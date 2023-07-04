import stanza.envs as envs
import jax
from jax.random import PRNGKey
from stanza.rl.ppo import PPO
from stanza.rl.nets import MLPActorCritic
from stanza.util.rich import StatisticsTable, ConsoleDisplay, LoopProgress

env = envs.create("pendulum")

net = MLPActorCritic(
    env.sample_action(PRNGKey(0))
)
params = net.init(PRNGKey(42), env.sample_state(PRNGKey(0)))

display = ConsoleDisplay()
display.add("ppo", StatisticsTable(), interval=1)
display.add("ppo", LoopProgress("RL"), interval=1)

with jax.profiler.trace("/tmp/jax-trace"):
    with display as dh:
        ppo = PPO()
        trained_params = ppo.train(
            PRNGKey(42),
            env, net.apply,
            params,
            rl_hooks=[dh.ppo]
        )