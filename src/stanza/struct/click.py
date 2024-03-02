from stanza import struct

from typing import List, Sequence, Tuple

import ast_comments
import ast
import inspect
import typing

import click
from click import *

import logging
logger = logging.getLogger(__name__)

def _option_from_field(field, prefix="", help=None):
    prefix = prefix or ""
    default = field.default
    required = field.required
    if field.type == typing.Optional[field.type]:
        base_type = typing.get_args(type)
        required = False
        default = None if default is struct.MISSING else default
    else:
        base_type = field.type
    yield FancyOption((f"--{prefix}{field.name}",),
        type=base_type, expose_value=False,
        default=default, required=required, help=help
    )

def _options_from_struct(type, prefix=""):
    docstrings = {f: v.help_string for (f,v) in _get_struct_docstrings(type).items()}
    for f in struct.fields(type):
        yield from _option_from_field(f, prefix=prefix, help=docstrings.get(f.name, None))

class FancyOption(click.Option):
    def __init__(self, param_decls : Sequence[str] | None = None,
                 type=None, struct_fields=True, include_struct_opt=True,
                 prefix_fields=True, **attrs):
        self.is_struct = struct.is_struct(type)
        self.include_struct_opt = include_struct_opt
        super().__init__(
            param_decls=param_decls,
            type=type,
            **attrs)
        self.sub_options = []

        if self.is_struct and struct_fields:
            prefix = click.parser.split_opt(self.opts[0])[1] + "." if prefix_fields else None
            self.sub_options.extend(_options_from_struct(type, prefix=prefix))
    
    def _parse_decls(self, decls: Sequence[str], expose_value: bool) -> tuple[str | list[str] | None]:
        name, opts, secondary_opts = super()._parse_decls(decls, expose_value)
        if self.is_struct and not self.include_struct_opt:
            opts = [], secondary_opts = []
        return name, opts, secondary_opts

    def get_help_record(self, ctx: click.Context) -> tuple[str, str] | None:
        if self.is_struct and not self.include_struct_opt:
            return None
        else:
            return super().get_help_record(ctx)

    def format_extended_help(self, ctx: click.Context, 
                             formatter: click.HelpFormatter) -> None:
        if self.sub_options:
            _format_options(self.sub_options, ctx, formatter)
    
    def add_to_parser(self, parser: click.OptionParser, ctx: click.Context) -> None:
        super().add_to_parser(parser, ctx)
        for sub_option in self.sub_options:
            sub_option.add_to_parser(parser, ctx)

def _format_options(params, ctx : click.Context, formatter : click.HelpFormatter):
    opts = [param.get_help_record(ctx) for param in params]
    opts = [("", "") if not opt else opt for opt in opts]
    from click.formatting import measure_table, iter_rows, term_len, wrap_text
    col_spacing = 2
    col_max = 30
    widths = measure_table(opts)
    first_col = min(widths[0], col_max) + col_spacing
    if len(widths) != 2:
        raise TypeError("Expected 2 columsn for options table")
    for (first, second), param in zip(
            iter_rows(opts, 2), params):
        # write the main option entry
        if first or second:
            formatter.write(f"{'':>{formatter.current_indent}}{first}")
            if not second:
                formatter.write("\n")
            else:
                if term_len(first) <= first_col - col_spacing:
                    formatter.write(" " * (first_col - term_len(first)))
                else:
                    formatter.write("\n")
                    formatter.write(" " * (first_col + formatter.current_indent))
                text_width = max(formatter.width - first_col - 2, 10)
                wrapped_text = wrap_text(second, text_width, preserve_paragraphs=True)
                lines = wrapped_text.splitlines()
                if lines:
                    formatter.write(f"{lines[0]}\n")
                    for line in lines[1:]:
                        formatter.write(f"{'':>{first_col + formatter.current_indent}}{line}\n")
                else:
                    formatter.write("\n")
        # if the option has sub-options, write them
        if hasattr(param, "format_extended_help"):
            param.format_extended_help(ctx, formatter)
    return True

class FancyCommand(click.Command):
    def format_options(self, ctx: Context, formatter: HelpFormatter) -> None:
        """Writes all the options into the formatter if they exist."""
        from click.formatting import measure_table, iter_rows, term_len
        params = self.get_params(ctx)
        opts = [param.get_help_record(ctx) for param in params]
        opts = [opt for opt in opts if opt is not None]
        if not opts:
            return
        _format_options(params, ctx, formatter)

def option(*param_decls, cls=None, type=None, **attrs):
    if cls is None:
        cls = FancyOption
    return click.option(*param_decls, cls=cls, 
                        type=type, **attrs)

def command(name=None, cls=None, **attrs):
    if cls is None:
        cls = FancyCommand
    return click.command(name=name, cls=cls, **attrs)

# def defaults_option(defaults, *param_decls, **attrs):
#     def defaults_decorator(f):
#         f = click.option(*param_decls, **attrs)(f)
#         f = struct.defaults(defaults)(f)
#         return f
#     return click.option(*param_decls, **attrs, cls=StructOption)

@struct.dataclass
class FieldDocString:
    comment_above: str = None
    comment_inline: str = None
    docstring_below: str = None
    desc_from_cls_docstring: str = None

    @property
    def help_string(self):
        return (self.docstring_below or self.comment_inline \
                or self.comment_above or self.desc_from_cls_docstring)

def _get_struct_docstrings(struct_cls, fields=None):
    fields = set(fields) if fields is not None else None
    if struct.is_struct_instance(struct_cls):
        struct_cls = struct_cls.__class__
    assert struct.is_struct(struct_cls)
    mro = inspect.getmro(struct_cls)
    assert mro[-1] is object
    mro = mro[:-1]

    docstrings = {}
    for base_class in mro:
        docs = _get_struct_src_docstrings(base_class, fields)
        if docs and fields is not None: # remove fields that have been documented
            fields = fields - set(docs.keys())
        docstrings.update(_get_struct_src_docstrings(base_class, fields))
        if fields is not None and not fields:
            break
    return docstrings

def _get_struct_src_docstrings(cls, fields):
    source = inspect.getsource(cls)
    if source is None:
        logger.warn(f"Could not get source for class {cls}")
        return None
    a = ast_comments.parse(source)
    cls_ast = a.body[0]
    assert isinstance(cls_ast, ast.ClassDef)
    docstrings = {}
    current_field = None
    current_docs = [None, None, None]
    def mk_field():
        nonlocal current_field, current_docs
        if current_field:
            docstrings[current_field] = FieldDocString(*current_docs)
            current_field = None
            current_docs = [None, None, None]

    for node in cls_ast.body:
        if isinstance(node, ast.AnnAssign):
            mk_field()
            current_field = node.target.id
        elif isinstance(node, ast_comments.Comment):
            if node.inline:
                current_docs[1] = node.value
            else:
                mk_field()
                current_docs[0] = node.value
        elif isinstance(node, ast.Expr) \
                and isinstance(node.value, ast.Constant) \
                    and isinstance(node.value.value, str):
            # string literal below is a "doc comment"
            current_docs[2] = (
                (current_docs[2] if current_docs[2] else "")
                    + node.value.value
            )
        else:
            mk_field()
    return docstrings