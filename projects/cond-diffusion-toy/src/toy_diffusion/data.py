from foundry.core import Array
from foundry.core.dataclasses import dataclass
from foundry.util.registry import Registry
from foundry.datasets.core import Dataset
from foundry.data import PyTreeData
from foundry.data.normalizer import Identity

import foundry.numpy as npx

@dataclass
class Sample:
    x: Array
    y: Array

@dataclass
class ThreeDeltas(Dataset[Sample]):
    def split(self, name):
        data = Sample(x=npx.array([0., 1., 1.]), y=npx.array([0.0, 0.5, 1.0]))
        return PyTreeData(data)
    
    def normalizer(self, name, **kwargs):
        if name == "identity":
            return Identity(
                Sample(x=npx.zeros(()), y=npx.zeros(()))
            )
        else:
            raise ValueError(f"Unknown normalizer {name}")

def make_three_deltas():
    return ThreeDeltas()

def register_all(registry : Registry):
    registry.register("three_deltas", make_three_deltas)