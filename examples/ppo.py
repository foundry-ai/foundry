import stanza.envs as envs
import stanza.policies as policies
import optax
import jax
from jax.random import PRNGKey
from stanza import Partial
from stanza.rl.ppo import PPO
from stanza.train import Trainer
from stanza.rl import ACPolicy
from stanza.rl.nets import MLPActorCritic
from stanza.util.loop import every_kth_iteration
from stanza.util.rich import StatisticsTable, ConsoleDisplay, LoopProgress

env = envs.create("pendulum")

net = MLPActorCritic(
    env.sample_action(PRNGKey(0))
)
params = net.init(PRNGKey(42),
    env.observe(env.sample_state(PRNGKey(0))))

display = ConsoleDisplay()
display.add("ppo", StatisticsTable(), interval=1)
display.add("ppo", LoopProgress("RL"), interval=1)

from stanza.reporting.wandb import WandbDatabase
db = WandbDatabase("dpfrommer-projects/examples").create()
from stanza.reporting.jax import JaxDBScope
db = JaxDBScope(db)

ppo = PPO(
    trainer = Trainer(
        optimizer=optax.chain(
            optax.clip_by_global_norm(0.5),
            optax.adam(3e-4, eps=1e-5)
        )
    )
)
with display as dh, db as db:
    logger_hook = db.statistic_logging_hook(
        log_cond=every_kth_iteration(1), buffer=100)
    trained_params = ppo.train(
        PRNGKey(42),
        env, net.apply,
        params,
        rl_hooks=[dh.ppo, logger_hook]
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