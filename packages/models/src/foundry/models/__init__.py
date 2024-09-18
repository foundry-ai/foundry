from foundry.util.registry import Registry

def register_all(registry: Registry, prefix=None):
    from . import resnet
    from . import mlp
    from . import gpt2
    from . import unet
    resnet.register(registry, prefix=prefix)
    mlp.register(registry, prefix=prefix)
    gpt2.register(registry, prefix=prefix)
    unet.register(registry, prefix=prefix)