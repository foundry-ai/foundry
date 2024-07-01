from stanza.util.registry import Registry, from_module

models = Registry()
models.extend("resnet", from_module(".resnet", "models"))

def create(name, /,  **kwargs):
    return models.create(name, **kwargs)