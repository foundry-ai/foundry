import jax
import jax.random
import tqdm

from collections import NamedTuple
from types import Any, List, Int

from jinx.random import PRNGSequence
from jinx.logging import logging_redirect_tqdm

class Vars:
    def __init__(self, vars):
        self.init_dataset = init_dataset
        self.vars = vars

class Optimizer:
    def __init__(self, vars, opt_init, opt_update):
        self._vars = vars
        self._opt_init = opt_init
        self._opt_update = opt_update

class Schedule:
    def __init__(self):
        self.items = []

    def add(self, item):
        self.items.append(item)
        return self
    
    def train(self, dataset, vars, optimizer, loss_fn, 
                quantity=Quantity.epochs(1)):
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