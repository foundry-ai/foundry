from stanza.util.attrdict import AttrDict
from stanza.util.dataclasses import dataclass, fields, Builder
from stanza.util.itertools import put_back, make_put_backable
from typing import List, Any
from enum import Enum
from functools import partial
import re
import importlib

def try_next(iter):
    try:
        return next(iter)
    except StopIteration:
        return None
    
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

        self.back_buffer = []
    
    def put_back(self, *tokens):
        tokens = list(tokens)
        tokens.reverse()
        self.back_buffer.extend(tokens)
    
    def __iter__(self):
        return self

    def __next__(self):
        if self.back_buffer:
            return self.back_buffer.pop()
        if not self.partial:
            self.partial = next(self.arg_iter)
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

class ArgParser:
    def default_context(self):
        raise NotImplementedError("default_context() not implemented")

    def parse_args(self, args, ctx=None):
        if ctx is None:
            ctx = self.default_context()
        tokens = ArgTokenizer(args)
        self.parse_tokens(tokens, ctx)
        token = try_next(tokens)
        if token:
            raise ArgParseError(f"Unexpected {token}")
        return ctx

def _setter(attr, obj, value):
    setattr(obj, attr, value)

class OptionParser(ArgParser):
    def __init__(self, long, short=None, type=None, populate=None, required=False):
        self.populate = populate or partial(_setter, long)
        if long.startswith('--'):
            long = long[2:]
        if short and short.startswith('-'):
            short = short[1:]
        self.required = required
        self.long = long
        self.short = short
        self.type = type
        self.expected = [Token(TokenType.OPTIONAL_LONG, long)]
        if self.short:
            self.expected.append(Token(TokenType.OPTIONAL_SHORT, short))
   
    def parse_tokens(self, tokens, ctx):
        opt = try_next(tokens)
        # No tokens for us to consume
        if not opt:
            return False
        if not ((opt.type == TokenType.OPTIONAL_LONG and opt.value == self.long) or \
                (self.short and opt.type == TokenType.OPTIONAL_SHORT and opt.value == self.short)):
            # This wasn't for us, put back the token and return False (no tokens consumed)
            put_back(opt, tokens)
            return False
        # If we have a type, consume the next argument
        if self.type is not None:
            next_token = try_next(tokens)
            if not next_token or (next_token.type != TokenType.ARG and next_token.type != TokenType.VALUE):
                raise ArgParseError("Expected value after f{opt}")
            value = self.type(next_token.value)
        else:
            value = True
        self.populate(ctx, value)
        return True
    
class PositionalParser(ArgParser):
    def __init__(self, populate, type=str, required=True):
        self.populate = populate
        self.required = required
    
    def parse_tokens(self, tokens, ctx):
        arg = try_next(tokens)
        if not arg: # If we are at the end
            return False
        if arg.type != TokenType.ARG:
            put_back(arg)
            return False
        self.populate(ctx, arg.value)
        return True

class MultiParser(ArgParser):
    def __init__(self, *parsers):
        self.parsers = parsers

    def parse_tokens(self, tokens, ctx):
        a = False
        done = False
        while not done:
            done = True
            for p in self.parsers:
                adv = p.parse_tokens(tokens, ctx)
                done = done and not adv
                a = a or adv
        return a


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
    
    def parse_tokens(self, tokens, ctx):
        consumed = False
        remaining_positional = list(self.positionals)
        remaining_positional.reverse()
        for t in tokens:
            put_back(tokens, t)
            if t.type  == TokenType.OPTIONAL_LONG:
                if not t.value in self.long_options:
                    break
                if not self.long_options[t.value].parse_tokens(tokens, ctx):
                    break
            elif t.type == TokenType.OPTIONAL_SHORT:
                if not t.value in self.short_options:
                    break
                if not self.short_options[t.value].parse_tokens(tokens, ctx):
                    break
            elif t.type == TokenType.ARG:
                p = remaining_positional.pop()
                if not p.parse_tokens(tokens, ctx):
                    break
            else:
                break
            consumed = True
        if remaining_positional and remaining_positional[0].required:
            raise ArgParseError("Required positional argument not specified")
        return consumed

class DataclassParser(SimpleParser):
    def __init__(self, dataclass):
        self.dataclass = dataclass
        super().__init__([OptionParser(long=f.name, type=f.type) for f in fields(dataclass)])
    
    def default_context(self):
        return Builder(self.dataclass)

# Allows for an option to be specified multiple times
class MultiOption(ArgParser):
    def __init__(self, dataclass):
        self.dataclass = dataclass
    
class MultiDataclassParser(SimpleParser):
    def __init__(self, dataclass):
        self.dataclass = dataclass
        super()._init__([DataclassParser.option])


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
    activity: Any # The activity entrypoint
    target: str
    database: str
    # TODO: Add dataset support
    configs: List[Any] # The config dataclasses

class RuntimeParser(ArgParser):
    def __init__(self):
        pass

    def default_context(self):
        return Builder(RuntimeConfig)
    
    def parse_tokens(self, tokens, cfg=None):
        opts = AttrDict()
        # Parse the generic tokens
        parser = SimpleParser([OptionParser("target", type=str),
                               OptionParser("database", type=str)],
                              [PositionalParser(partial(_setter, "entrypoint"))])
        parser.parse_tokens(tokens, opts)
        # Set defaults
        opts.target = opts.get("target", None) or "poetry://localhost"
        opts.database = opts.get("database", None) or None
        opts.entrypoint = opts.get("entrypoint", None) or None

        activity = load_entrypoint(opts.entrypoint)

        cfg.activity = activity
        cfg.target = opts.target
        cfg.database = opts.database

        dc_parser = DataclassParser(activity.config_dataclass)
        dc_builder = Builder(activity.config_dataclass)
        dc_parser.parse_tokens(tokens, dc_builder)

        cfg.configs = [dc_builder.build()]
        return cfg