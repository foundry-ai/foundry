from dataclasses import dataclass
from stanza.util import ThreadLocalStack
from typing import Set

import jax.tree_util
import weakref

__context_stack: ThreadLocalStack["Context"] = ThreadLocalStack()

def current_context():
    return __context_stack.peek(default=None)


@dataclass
class ModuleNode:
    pass

class Module:
    def __init__(self, *, name=None):
        ctx = current_context()
        if ctx is not None:
            self.__mod_weak_parent = weakref.ref(ctx.current_module)
        else:
            self.__mod_weak_parent = None
        self.__mod_parent = None
    
    def __tree__(self):
        pass

    def __node__(self, parent=None):

# turns a tree of modules into a
def lift_tree(tree, parent=None):
    def map_fn(x):
        if not isinstance(x, Module): return x
    tree = jax.tree_util.tree_map(map_fn, tree, is_leaf=lambda x: isinstance(x, Module))
    for c in children:
        if isinstance(c, Module):
            lift_tree(c, paren)
    children = [map_child(child) for child in children]
    print(children)

# A "lifting context"
@dataclass
class Context:
    current_module: Module
    mutated_inputs: Set

    # mutate an object
    def mutate(self, key, new_value):
        self.objects[key] = new_value

    def __enter__(self):
        __context_stack.push(self)

    def __exit__(self, *args):
        __context_stack.pop()
