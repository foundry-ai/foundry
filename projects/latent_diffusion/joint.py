import optax
import jax
import jax.numpy as jnp

from stanza.dataclasses import dataclass
from stanza.runtime import activity
from stanza import partial
from stanza.data import PyTreeData
from stanza.train import batch_loss, express as express_trainer
from stanza.train.validate import Generator
from stanza.util.random import PRNGSequence
import stanza.util.loop as loop
from stanza.util import l2_norm_squared
from stanza.diffusion.ddpm import DDPMSchedule

from latent_diffusion.nets import make_joint_network
from latent_diffusion.data import load_data

@dataclass(jax=True)
class Config:
    dataset : str = "cifar10"
    latent_dim : int = 1
    learning_rate : float = 1e-3
    epochs : int = 50
    batch_size : int = 64
    timesteps : int = 100
    rng_seed : int = 42

def sample_conditionals(rng_key, net, fn_params, schedule, x_sample, z_sample, 
                        x_timestep, z_timestep, train=False):
    def z_cond_x_model(rng, sample, t):
        joint_sample = (x_sample, sample)
        _, s = net.apply(fn_params, joint_sample,
                  timestep=x_timestep, latent_timestep=t,
                  rngs={"dropout":rng}, train=train)
        return s
    def x_cond_z_model(rng, sample, t):
        joint_sample = (sample, z_sample)
        s, _ = net.apply(fn_params, joint_sample,
                     timestep=t, latent_timestep=z_timestep,
                     rngs={"dropout":rng}, train=train)
        return s
    x_cond = schedule.sample(rng_key, x_cond_z_model, x_sample,
                             final_step=x_timestep, static_loop=True)
    z_cond = schedule.sample(rng_key, z_cond_x_model, z_sample,
                             final_step=z_timestep, static_loop=True)
    return x_cond, z_cond

def loss_fn(config, schedule, normalizer, net,
            fn_state, fn_params, 
            rng_key, sample):
    rng = PRNGSequence(rng_key)
    dist_x_timestep = 0
    dist_z_timestep = 0
    # dist_x_timestep = jax.random.randint(next(rng), (), 0, schedule.num_steps)
    # dist_z_timestep = jax.random.randint(next(rng), (), 0, 1)
    sub_x_timestep = jax.random.randint(next(rng), (), dist_x_timestep, schedule.num_steps)
    sub_z_timestep = jax.random.randint(next(rng), (), dist_z_timestep, schedule.num_steps)

    normalized_sample = normalizer.normalize(sample)
    latent_sample = jax.random.truncated_normal(next(rng), -1., 1., (config.latent_dim,))
    x, _, _ = schedule.add_noise(next(rng), normalized_sample, dist_x_timestep)
    z, _, _ = schedule.add_noise(next(rng), latent_sample, dist_z_timestep)
    # x, z = normalized_sample, latent_sample

    # we now have samples along the x and z chains
    # now we conditionally integrate the x along the z direction
    # and the z along the x direction to get two joint samples
    # (x,z) at the (x_timestep, z_timestep) location. We use the sample
    # from the x chain as the target for the z chain and vice versa
    # to get the joint distributions to match
    alt_x, alt_z = sample_conditionals(next(rng), net, fn_params, schedule, x, z,
                                       schedule.num_steps - dist_x_timestep, 
                                       schedule.num_steps - dist_z_timestep, train=True)
    # alt_x, alt_z = noisy_x, noisy_z
    # we now have (alt_x, noisy_z) and (alt_z, noisy_x) 
    # samples at the (x_timestep, z_timestep) timestep location
    # generate sub-timesteps sub_x_timestep, and sub_z_timestep
    (sub_a_x, sub_b_x), _, (target_a_x, target_b_x) = schedule.add_sub_noise(
        next(rng), (alt_x, x),
        dist_x_timestep, sub_x_timestep
    )
    (sub_a_z, sub_b_z), _, (target_a_z, target_b_z) = schedule.add_sub_noise(
        next(rng), (z, alt_z),
        dist_z_timestep, sub_z_timestep
    )
    sub_a, sub_b = (sub_a_x, sub_a_z), (sub_b_x, sub_b_z)
    target_a, target_b = (target_a_x, target_a_z), (target_b_x, target_b_z)

    # a has the alt x, true latent
    # b has the alt latent, true x

    pred_a = net.apply(fn_params, sub_a, 
            timestep=sub_x_timestep, latent_timestep=sub_z_timestep, train=True)
    pred_b = net.apply(fn_params, sub_b, 
            timestep=sub_x_timestep, latent_timestep=sub_z_timestep, train=True)
    loss_a = l2_norm_squared(pred_a, target_a)
    loss_b = l2_norm_squared(pred_b, target_b)
    # weight the alt latent, true x process higher
    loss = loss_a + loss_b
    stats = { "loss_a": loss_a, "loss_b": loss_b, "loss": loss }
    return fn_state, loss, stats

@activity(Config)
def train(config, db):
    rng = PRNGSequence(config.rng_seed)
    exp = db.create()

    data = load_data(config.dataset, next(rng))
    normalizer = data.normalizer
    # make the network
    data_sample = data.train_data[0]
    latent_sample = jnp.zeros((config.latent_dim,))
    net, params = make_joint_network(next(rng), data_sample, latent_sample)

    schedule = DDPMSchedule.make_squaredcos_cap_v2(
        config.timesteps,
        clip_sample_range=1.
    )
    loss = batch_loss(partial(loss_fn, config, schedule, normalizer, net))

    steps = config.epochs * (data.train_data.length // config.batch_size)
    optimizer = optax.adamw(
        optax.cosine_decay_schedule(config.learning_rate, steps),
    )

    def do_generate(train_state, rng_key):
        fn_params = train_state.fn_params
        model = lambda rng, sample, t: net.apply(
            fn_params, sample, timestep=t, latent_timestep=t, rngs={"dropout":rng}
        )
        sample, latent = schedule.sample(rng_key, model,
                                        (data_sample, latent_sample))
        sample = normalizer.unnormalize(sample)
        return sample, latent
    
    def do_visualize(joint_samples):
        samples, latents = joint_samples
        return data.samples_visualizer(samples, latents=latents)

    # log some ground truth samples
    test_samples = PyTreeData.from_data(data.test_data[:data.n_visualize_samples]).data
    exp.log({"test_samples": data.samples_visualizer(test_samples)})

    generator = Generator(
        bucket=exp,
        rng_key=next(rng),
        samples=data.n_visualize_samples,
        generate_fn=do_generate,
        visualizer=do_visualize
    )
    trainer = express_trainer(
        optimizer=optimizer,
        batch_size=config.batch_size,
        max_epochs=config.epochs,
        validate_condition=loop.every_epoch,
        validate_dataset=data.test_data,
        validate_batch_size=config.batch_size,
        validate_rng=next(rng),
        bucket=exp,
        loss_fn=loss,
        train_hooks=[generator]
    )
    res = trainer.train(
        data.train_data,
        init_params=params,
        rng_key=next(rng)
    )
    return res
