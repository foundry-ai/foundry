import inspect
import importlib
import functools

def builder_factory(module_name, func_name):
    frm = inspect.stack()[1]
    pkg = inspect.getmodule(frm[0]).__name__
    def make_builder():
        mod = importlib.import_module(module_name, package=pkg)
        return getattr(mod, func_name)
    return make_builder

_BUILDER_FACTORIES = {
    "mnist": builder_factory(".mnist", "mnist"),
    "cifar10": builder_factory(".cifar", "cifar10"),
    "cifar100": builder_factory(".cifar", "cifar100"),
    "celeb_a": builder_factory(".celeb_a", "celeb_a") 
}

def load(name, **kwargs):
    """Load a dataset by name."""
    return _BUILDER_FACTORIES[name]()(**kwargs)

def builder(func):
    @functools.wraps(func)
    def wrapper(*, splits=set(), **kwargs):
        if isinstance(splits, str):
            split_set = {splits}
            data = func(splits=split_set, **kwargs)
            return data[splits]
        if not isinstance(splits, set):
            split_set = set(splits)
            data = func(splits=split_set, **kwargs)
            res = tuple((data[k] for k in splits))
            return res
        else:
            return func(splits=splits, **kwargs)
    return wrapper