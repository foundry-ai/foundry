import stanza.envs as envs
import stanza.policies as policies
import optax
import jax
from jax.random import PRNGKey
from stanza import Partial
from stanza.rl.ppo import PPO
from stanza.train import Trainer
from stanza.rl import EpisodicEnvironment, ACPolicy
from stanza.rl.nets import MLPActorCritic
from stanza.util.rich import StatisticsTable, ConsoleDisplay, LoopProgress

env = envs.create("pendulum")
# will automatically reset when done
# or when 1000 timesteps have been reached
env = EpisodicEnvironment(env, 1000)

net = MLPActorCritic(
    env.sample_action(PRNGKey(0))
)
params = net.init(PRNGKey(42),
    env.observe(env.sample_state(PRNGKey(0))))

display = ConsoleDisplay()
display.add("ppo", StatisticsTable(), interval=1)
display.add("ppo", LoopProgress("RL"), interval=1)

ppo = PPO(
    trainer = Trainer(
        optimizer=optax.chain(
            optax.clip_by_global_norm(0.5),
            optax.adam(3e-4, eps=1e-5)
        )
    )
)

with display as dh:
    trained_params = ppo.train(
        PRNGKey(42),
        env, net.apply,
        params,
        rl_hooks=[dh.ppo]
    )

ac_apply = Partial(net.apply, trained_params.fn_params)
policy = ACPolicy(ac_apply)

r = policies.rollout(env.step, 
    env.reset(PRNGKey(42)), policy, 
    model_rng_key=PRNGKey(31231),
    policy_rng_key=PRNGKey(43232),
    observe=env.observe,
    length=200)

print(jax.vmap(env.observe)(r.states))