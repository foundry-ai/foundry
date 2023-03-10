from stanza.util.attrdict import AttrDict
from stanza.util.dataclasses import dataclass, field, fields, Parameters
from stanza.util.itertools import put_back, make_put_backable
from typing import List, Any
from enum import Enum
from functools import partial
import itertools
import re
import importlib

    
# -------------- A lightweight argparse alternative ------------
# This exposes an extensible set of argument parser utilities
# which can be composed to handle arbitrarily complex
# argument structures. It is particularly useful
# because it can be used to

class TokenType(Enum):
    ARG = 1
    VALUE = 2 # token after the = after a --option=
    OPTIONAL_LONG = 3
    OPTIONAL_SHORT = 4
    LBRAKET = 5
    RBRAKET = 6
    # A standalone -- character
    SEPARATOR = 7

@dataclass(jax=False)
class Token:
    type: TokenType
    value: str = None

    def __repr__(self):
        return '(' + self.type.name + ',' + self.value + ')'

class ArgParseError(ValueError):
    def __init__(self, msg):
        super().__init__(msg)

# An argv tokenizer
class ArgTokenizer:
    def __init__(self, arg_iter):
        self.arg_iter = iter(arg_iter)
        # If we just parsed a --long_opt or -short_opt
        # ignore any = following immediately after
        self.after_opt = False
        self.partial = None

        self.consumed = 0
        self.back_buffer = []
    
    def put_back(self, *tokens):
        tokens = list(tokens)
        tokens.reverse()
        self.consumed = self.consumed - len(tokens)
        self.back_buffer.extend(tokens)
    
    def __iter__(self):
        return self

    def __next__(self):
        if self.back_buffer:
            self.consumed = self.consumed + 1
            return self.back_buffer.pop()
        if not self.partial:
            self.partial = next(self.arg_iter)
        self.consumed = self.consumed + 1
        # If after an option in the same token
        # spaces, newlines, or = signs that follow
        # and then turn the remaining token as an argument
        # this way something like
        # --extra_args="--bar --bar" will parse properly
        if self.after_opt and self.partial:
            arg = self.partial.lstrip('= \t\n\r')
            # still 
            self.partial = None
            self.after_opt = False
            return Token(TokenType.VALUE, arg)
        
        proc = self.partial.lstrip('= \t\n\r')

        # If we get just a -- this is a SEPARATOR token 
        if proc.strip() == '--':
            self.partial = None
            return Token(TokenType.SEPARATOR)
        # Parse either --opt or -opt
        elif proc.startswith('-'):
            rest, type = (proc[2:], TokenType.OPTIONAL_LONG) \
                    if proc.startswith('--') else \
                        (proc[1:], TokenType.OPTIONAL_SHORT)
            # find until = if it exists
            match = re.search(r" |\t|\n|=", rest)
            if match:
                span = match.span()
                arg, rest = rest[:span[0]], rest[span[1]:]
                if rest.startswith('='):
                    self.after_opt = True
            else:
                arg = rest
                rest = None
            self.partial = rest
            return Token(type, arg)
        
        # It is just a positional argument
        else:
            arg = self.partial
            self.partial = None
            if arg == '[':
                return Token(TokenType.LBRAKET)
            elif arg == ']':
                return Token(TokenType.RBRAKET)
            return Token(TokenType.ARG, arg)

# --- Iterator utiltiies -----
def try_next(iter):
    try:
        return next(iter)
    except StopIteration:
        return None

def peek(iter):
    try:
        n = next(iter)
        put_back(iter, n)
        return n
    except StopIteration:
        return None

# Consume all tokens of particular types
def consume_tokens(iter, *types):
    for t in iter:
        if t.type in types:
            yield t
        else:
            # put back the token and 
            # return
            put_back(iter, t)
            break

class ArgParser:
    def default_context(self):
        raise NotImplementedError("default_context() not implemented")

    def parse_args(self, args, ctx=None):
        if ctx is None:
            ctx = self.default_context()
        tokens = ArgTokenizer(args)
        ctx = self.parse_tokens(ctx, tokens)
        token = try_next(tokens)
        if token:
            raise ArgParseError(f"Unexpected {token}")
        return ctx

class OptionParser(ArgParser):
    def __init__(self, long, callback, short=None, nargs=0):
        self.callback = callback
        if long.startswith('--'):
            long = long[2:]
        if short and short.startswith('-'):
            short = short[1:]
        self.long = long
        self.short = short
        self.nargs = nargs
   
    def parse_tokens(self, ctx, tokens):
        opt = try_next(tokens)
        # No tokens for us to consume
        if not opt:
            return
        if not ((opt.type == TokenType.OPTIONAL_LONG and opt.value == self.long) or \
                (self.short and opt.type == TokenType.OPTIONAL_SHORT and opt.value == self.short)):
            # This wasn't for us, put back the token and return False (no tokens consumed)
            put_back(opt, tokens)
            return ctx
        # If we have a type, consume the next argument
        if self.nargs != 0:
            if self.nargs == '+':
                arg_tokens = [t for t in consume_tokens(tokens, 
                                TokenType.ARG, TokenType.VALUE)]
            else:
                arg_tokens = list(itertools.islice(tokens, self.nargs))
                for t in arg_tokens:
                    if t.type != TokenType.ARG and t.type != TokenType.VALUE:
                        raise ArgParseError("Expected more option arguments")
            r = self.callback(ctx, *[t.value for t in arg_tokens])
            ctx = r if r is not None else ctx
        else:
            r = self.callback(ctx)
            ctx = r if r is not None else ctx
        return ctx

class PositionalParser(ArgParser):
    def __init__(self, callback, nargs=1):
        assert nargs > 0 or nargs == '+'
        self.nargs = nargs
        self.callback = callback

    def parse_tokens(self, ctx, tokens):
        if self.nargs == '+':
            arg_tokens = [t for t in consume_tokens(tokens, 
                            TokenType.ARG, TokenType.VALUE)]
        else:
            arg_tokens = list(itertools.islice(tokens, self.nargs))
            for t in arg_tokens:
                if t.type != TokenType.ARG and t.type != TokenType.VALUE:
                    raise ArgParseError("Expected more option arguments")
        args = [t.value for t in arg_tokens]
        r = self.callback(ctx, *args)
        ctx = r if r is not None else ctx
        return ctx

class MultiParser(ArgParser):
    def __init__(self, *parsers):
        self.parsers = parsers

    def parse_tokens(self, ctx, tokens):
        while True:
            pos = tokens.consumed
            for p in self.parsers:
                ctx = p.parse_tokens(ctx, tokens)
            # If we weren't able to consume
            # any more tokens, just break
            if tokens.consumed == pos:
                break
        return ctx

# A SimpleParser handles options and positionals in the standard manner
class SimpleParser(ArgParser):
    def __init__(self, options, positionals=None):
        self.positionals = positionals or []
        self.options = options
        self.long_options = {
            o.long: o for o in options
        }
        self.short_options = {
            o.short: o for o in options if o.short
        }
    
    def parse_tokens(self, ctx, tokens):
        remaining_positional = list(self.positionals)
        remaining_positional.reverse()
        for t in tokens:
            put_back(tokens, t)
            pos = tokens.consumed
            if t.type  == TokenType.OPTIONAL_LONG:
                if not t.value in self.long_options:
                    break
                ctx = self.long_options[t.value].parse_tokens(ctx, tokens)
            elif t.type == TokenType.OPTIONAL_SHORT:
                if not t.value in self.short_options:
                    break
                ctx = self.short_options[t.value].parse_tokens(ctx, tokens)
            elif t.type == TokenType.ARG:
                if not remaining_positional:
                    raise ArgParseError("Unexpected positional argument")
                p = remaining_positional.pop()
                ctx = p.parse_tokens(ctx, tokens)
            # We are no longer able to consume tokens
            if tokens.consumed == pos:
                break
        return ctx

# Allows for parsing sets of parameters (see Parameters from stanza.util.dataclasses module)
class ParametersParser(SimpleParser):
    def __init__(self, dataclass, multi_parser=False):
        self.dataclass = dataclass
        self.multi_parser = multi_parser
        if multi_parser:
            super().__init__([ParametersParser.make_field_multi_option(dataclass, f) \
                        for f in fields(dataclass)])
        else:
            super().__init__([ParametersParser.make_field_option(dataclass, f) \
                        for f in fields(dataclass)])

    # def default_context(self):
    #     return {Builder(self.dataclass)} if self.multi_parser else Builder(self.dataclass)

    @staticmethod
    def make_field_option(dc, field):
        def setter(cfg, arg):
            v = field.type(arg)
            return cfg.set(field.name, v)
        return OptionParser(field.name, setter, nargs=1)

    @staticmethod
    def make_field_multi_option(dc, field):
        def setter(curr_parameters, *args):
            if len(args) == 0:
                raise ArgParseError("Expected at least one argument")
            vs = [field.type(a) for a in args]
            parameters = {Parameters(**{field.name: v}) for v in vs}
            new_parameters = Parameters.cartesian_product(curr_parameters, parameters)
            # replace curr_parameters with new_parameters
            return new_parameters
        return OptionParser(field.name, setter, nargs='+')

def load_entrypoint(entrypoint_string):
    parts = entrypoint_string.split(":")
    if len(parts) != 2:
        raise ValueError("Entrypoint must include module and activity")
    module, attr = parts
    module = importlib.import_module(module)
    return getattr(module, attr)

# -------- Utilities for parsing config options ----------
@dataclass
class RuntimeConfig:
    activity: Any = None # The activity entrypoint
    target: str = None
    database: str = None
    # TODO: Add dataset support
    configs: List[Any] = field(default_factory=list)

class RuntimeParser(ArgParser):
    def __init__(self):
        pass

    def default_context(self):
        return RuntimeConfig()
    
    def parse_tokens(self, cfg, tokens):
        opts = AttrDict()
        # Parse the generic tokens
        def setter(param, opts, val):
            opts[param] = val
        parser = SimpleParser([OptionParser("target", partial(setter, "target"), nargs=1),
                               OptionParser("db", partial(setter, "database"), nargs=1)],
                              [PositionalParser(partial(setter, "entrypoint"))])
        parser.parse_tokens(opts, tokens)
        # Set defaults
        opts.target = opts.get("target", None) or "poetry://localhost"
        opts.database = opts.get("database", None) or "dummy://"
        opts.entrypoint = opts.get("entrypoint", None) or None
        if opts.entrypoint is None:
            raise ArgParseError("Must specify entrypoint")

        activity = load_entrypoint(opts.entrypoint)

        cfg.activity = activity
        cfg.target = opts.target
        cfg.database = opts.database

        dc_parser = ParametersParser(activity.config_dataclass, multi_parser=True)
        dc_builders = {Parameters()}
        dc_builders = dc_parser.parse_tokens(dc_builders, tokens)
        cfg.configs = [b(activity.config_dataclass) for b in dc_builders]

        # cfg.configs = [activity.config_dataclass()]
        return cfg