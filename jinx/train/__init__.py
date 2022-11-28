import jax
import jax.random
import tqdm

from collections import NamedTuple
from types import Any, List, Int

from jinx.random import PRNGSequence
from jinx.dataset import INFINITE
from jinx.logging import logging_redirect_tqdm

class Optimization:
    def __init__(self, fun, vars_init, opt_init, opt_update):
        self._fun = fun
        self._vars = vars_init
        self._loss_fn = loss_fn
        self._opt_init = opt_init
        self._opt_update = opt_update

class DataSource:
    def __init__(self, dataset, shuffle_key, shuffle_buffer_size):
        shuffle_rng, shuffle_key = jax.random.split(shuffle_key)
        self._dataset = dataset
        self._shuffle_key = shuffle_key

        self._shuffle_buffer_size = shuffle_buffer_size
        self._shuffled_dataset = dataset.shuffle(shuffle_rng, shuffle_buffer_size)
        self._iterator = self._shuffled_dataset.start

        self._samples_per_epoch = self._dataset.remaining(self._dataset.start)
        self._current_epoch = 0

    @property
    def current_iterator(self):
        return self._iterator

    @property
    def current_dataset(self):
        return self._shuffled_dataset
    
    def update_iterator(self, new_iterator):
        # update the iterator
        if self.current_dataset.is_end(new_iterator):
            # re-shuffle
            shuffle_rng, shuffle_key = jax.random.split(shuffle_key)
            self._shuffle_key = shuffle_key
            self._shuffled_dataset = dataset.shuffle(shuffle_rng, self._shuffle_buffer_size)
            self._current_epoch = self._current_epoch + 1
        else:
            self._current = new_iterator

class Schedule:
    def __init__(self):
        self.items = []

    def add(self, item):
        self.items.append(item)
        return self
    
    def train(self, optimization, dataset, batch_size, epochs, steps=0):
        train = Train(dataset, vars, optimizer, loss_fn, quantity)
        self.items.append(train)
        return train

    def test(self, test_fn, duration=Quantity.iterations(1)):
        test = TestItem(test_fn, duration)
        self.items.append(test)
        return test

    def __call__(self):
        while True:
            for i in self.items:
                i()

class Train:
    def __init__(self, dataset, vars, optimizer, loss_fn, quantity):
        self.quantity = quantity
        self.loss_fn = loss_fn

    def run(self, state):
        # if the optimizer state has not yet been
        pass

class TestItem:
    def __init__(self, loss_fn, duration):
        self.duration = duration
        self.loss_fn = loss_fn

    def run(self, dataset, iterator):
        pass

class Trainer:
    def __init__(self, dataset, loss_fn, validate_fn, opt_update, epochs, 
                    shuffle_buffer_size=None):
        self._epochs = epochs
        self._dataset = dataset
        self._loss_fn = loss_fn
        self._opt_update = opt_update
        self._shuffle_buffer_size = shuffle_buffer_size or 1024

        self._report_train_stats_fn = None
        self._report_test_stats_fn = None

    def _report_stats_fn(self, stats):
        pass

    def train_epoch(self, shuffled_data, rng_key, opt_state, params):
        with tqdm.tqdm(total=dataset.length,leave=False) as ibar:
            iterator = dataset.start
        pass

    def train(self, rng_key, opt_state, params):
        shuffle_key, train_key = jax.random.split(rng_key)
        shuffle_rng = PRNGSequence(shuffle_key)
        train_rng = PRNGSequence(train_key)

        with logging_redirect_tqdm():
            with tqdm.tqdm(total=self._epochs) as pbar:
                for epoch in range(self._epochs):
                    dataset = self._dataset.shuffle(next(shuffle_rng), self._shuffle_buffer_size)


                    pbar.update(1)