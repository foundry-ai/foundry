import jax
import tqdm


class Trainer:
    # loss_fn takes arguments (params, state, element)
    def __init__(self, optax_transform, loss_fn, batch_size):
        self.optax_transform = optax_transform
        self.loss_fn = loss_fn
        self.batch_size = batch_size
    
    def _train_epoch(self, dataset, rng_key,
                    init_params, init_state=None,
                    iterations):
        rng_key, sk = jax.random.split(rng_key)
        # shuffle the dataset
        dataset = dataset.shuffle(sk)
    
    def train(self, dataset, rng_key,
                init_params, init_state=None,
                epochs=1, iterations=None):