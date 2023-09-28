from stanza.runtime import activity
from stanza.dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import optax
from stanza.data import Data, PyTreeData
from stanza.data.cifar import cifar10
from stanza.train import Trainer, SAMTrainer, batch_loss
from stanza import partial
from stanza.util.logging import logger
from stanza.util.random import PRNGSequence
from stanza.util.rich import ConsoleDisplay, LoopProgress, EpochProgress, StatisticsTable

@dataclass
class Config:
    net: str = "resnet18"
    skip_connections: bool = True
    sam_epochs: int = None
    normalize: bool = True
    use_sam: bool = True
    sam_resample: bool = True
    epochs: int = 20
    seed: int = 42
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    rho: float = 0.05
    label_noise : float = 0.0

def accuracy(model, vars, sample):
    x, y = sample
    y_logits = model.apply(vars, x, train=False)
    y_logits = jax.nn.log_softmax(y_logits)
    y_pred = jnp.argmax(y_logits, axis=-1)
    y_true = jnp.argmax(y, axis=-1)
    return 1*(y_pred == y_true)

def loss_fn(model, batch_stats, params, rng_key, sample):
    x, y = sample
    return batch_stats, 0., {"loss": 0.}
    vars = {
        "batch_stats": batch_stats,
        "params": params
    }
    y_logits, updates = model.apply(vars, x, mutable=['batch_stats'])
    y_logits = jax.nn.log_softmax(y_logits)
    batch_stats = updates["batch_stats"]
    loss = -jnp.mean(jnp.sum(y_logits * y, axis=-1))
    stats = {"loss": loss}
    return batch_stats, loss, stats

@activity(Config)
def train(config, db):
    logger.info("Running: {}", config)
    logger.info("Loading data...")
    train_data, test_data = make_data(config)
    rng = PRNGSequence(config.seed)

    if config.sam_resample:
        batch_size = config.batch_size * 2
    else:
        batch_size = config.batch_size

    # sam_iterations = (config.sam_epochs * \
    #              (train_data.length // config.batch_size)) \
    #             if config.sam_epochs is not None else None

    model = make_net(config)
    # use first sample to initialize model
    logger.info("Initializing model...")
    sample_batch = train_data.sample_batch(batch_size, next(rng))
    init_vars = jax.jit(model.init)(next(rng), sample_batch[0])

    init_params = init_vars["params"]
    batch_stats = init_vars["batch_stats"]

    loss = batch_loss(partial(loss_fn, model))

    steps = config.epochs * (train_data.length // batch_size)
    optimizer = optax.adamw(optax.cosine_decay_schedule(config.lr, steps),
                            weight_decay=config.weight_decay)
    sub_optimizer = optax.scale(config.rho)

    if config.use_sam:
        trainer = SAMTrainer(
            max_epochs=config.epochs,
            batch_size=batch_size,
            optimizer=optimizer,
            sub_optimizer=sub_optimizer,
            normalize=config.normalize)
    else:
        trainer = Trainer(
            max_epochs=config.epochs,
            batch_size=batch_size,
            optimizer=optimizer)

    display = ConsoleDisplay()
    display.add("train", StatisticsTable(), interval=100)
    display.add("train", LoopProgress(), interval=100)
    display.add("train", EpochProgress(), interval=100)

    logger.info("Training...")
    with display as w:
        res = trainer.train(
            train_data,
            loss_fn=loss,
            rng_key=next(rng),
            init_params=init_params,
            init_state=batch_stats,
            hooks=[w.train]
        )

    params, stats = res.fn_params, res.fn_state
    vars = {
        "params": params,
        "batch_stats": stats
    }
    acc = jnp.mean(jax.vmap(accuracy, in_axes=(None, None, 0))(model, vars, test_data.data))
    logger.info("Test Accuracy: {}", acc)

def make_data(config):
    train_data, test_data = cifar10()
    def map_fn(sample):
        x, y = sample
        x = x.astype(float) / 255
        y = jax.nn.one_hot(y, 10)
        return x, y
    train_data = PyTreeData.from_data(train_data.map(map_fn))
    test_data = PyTreeData.from_data(test_data.map(map_fn))
    return train_data, test_data

def make_net(config):
    from stanza.nets.resnet import \
        ResNet9, ResNet18, ResNet34, ResNet50
    if config.net == "resnet9":
        net = ResNet9
        net = partial(net, skip_connections=config.skip_connections)
    elif config.net == "resnet18":
        net = ResNet18
        net = partial(net, skip_connections=config.skip_connections)
    elif config.net == "resnet34":
        net = ResNet34
        net = partial(net, skip_connections=config.skip_connections)
    elif config.net == "resnet50":
        net = ResNet50
        net = partial(net, skip_connections=config.skip_connections)
    return net(num_classes=10)