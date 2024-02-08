from stanza.struct import fields

def click_options_from_struct(struct):
    default = struct()
    import click
    def _(function):
        for field in fields(struct):
            name = field.name.replace("_", "-")
            if field.type == bool:
                function = click.option(
                    f"--{name}/--no-{name}",
                    default=getattr(default, field.name)
                )(function)
            else:
                function = click.option(
                    f"--{name}", type=field.type,
                    default=getattr(default, field.name)
                )(function)
        return function
    return _