from stanza import struct
import inspect
import ast
import ast_comments

import logging
logger = logging.getLogger(__name__)

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

def get_struct_docstrings(struct_cls, fields=None):
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
                current_docs[1] = node.value.lstrip("# ")
            else:
                mk_field()
                current_docs[0] = node.value.lstrip("# ")
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