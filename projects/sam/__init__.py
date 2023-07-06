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
from stanza.util.rich import ConsoleDisplay, LoopProgress, StatisticsTable

@dataclass
class Config:
    net: str = "resnet18"
    skip_connections: bool = True
    normalize: bool = True
    use_sam: bool = False
    epochs: int = 100
    batch_size: int = 128

def accuracy(model, vars, sample):
    x, y = sample
    y_logits = model.apply(vars, x, train=False)
    y_logits = jax.nn.log_softmax(y_logits)
    y_pred = jnp.argmax(y_logits, axis=-1)
    y_true = jnp.argmax(y, axis=-1)
    return 1*(y_pred == y_true)

def loss_fn(model, batch_stats, params, rng_key, sample):
    x, y = sample
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
    train_data, test_data = make_data(config)
    rng = PRNGSequence(42)

    model = make_net(config)
    # use first sample to initialize model
    sample_batch = train_data.sample_batch(config.batch_size, next(rng))
    init_vars = model.init(next(rng), sample_batch[0])

    init_params = init_vars["params"]
    batch_stats = init_vars["batch_stats"]

    loss = batch_loss(partial(loss_fn, model))

    optimizer = optax.adam(1e-3)
    sub_optimizer = optax.scale(1e-3)

    if config.use_sam:
        trainer = SAMTrainer(
            epochs=config.epochs,
            batch_size=config.batch_size,
            optimizer=optimizer,
            sub_optimizer=sub_optimizer,
            normalize=config.normalize)
    else:
        trainer = Trainer(
            epochs=config.epochs,
            batch_size=config.batch_size,
            optimizer=optimizer)

    display = ConsoleDisplay()
    display.add("train", StatisticsTable(), interval=1)
    display.add("train", LoopProgress(), interval=1)

    with display as w:
        res = trainer.train(
            loss,
            train_data,
            next(rng),
            init_params,
            batch_stats,
            hooks=[w.train]
        )

    params, stats = res.fn_params, res.fn_state
    vars = {
        "params": params,
        "batch_stats": stats
    }
    accuracy = jnp.mean(jax.vmap(accuracy, in_axes=(None, None, 0))(model, vars, test_data.data))
    logger.info("Test Accuracy: {}", accuracy)

@dataclass
class SweepConfig:
    pass

@activity(SweepConfig)
def sweep(sweep_config, db):
    pass

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
        ResNet18, ResNet34, ResNet50
    if config.net == "resnet18":
        net = ResNet18
    elif config.net == "resnet34":
        net = ResNet34
    elif config.net == "resnet50":
        net = ResNet50
    net = partial(net, skip_connections=config.skip_connections)
    return net(num_classes=10)