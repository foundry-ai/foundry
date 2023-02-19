from stanza.util.dataclasses import dataclass
from typing import List, Any
from enum import Enum
import re



# -------------- A lightweight argparse alternative ------------
# This exposes an extensible set of argument parser utilities
# which can be composed to handle arbitrarily complex
# argument structures. It is particularly useful
# because it can be used to

class TokenType(Enum):
    ARG = 1
    OPTIONAL_LONG = 2
    OPTIONAL_SHORT = 3
    LBRAKET = 4
    RBRAKET = 5
    # A standalone -- character
    SEPARATOR = 6

@dataclass(jax=False)
class Token:
    type: TokenType
    value: str = None

    def __repr__(self):
        return '(' + self.type.name + ',' + self.value + ')'

class ArgError(ValueError):
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
    
    def __iter__(self):
        return self

    def __next__(self):
        if not self.partial:
            self.partial = next(self.arg_iter)
        # If after an option in the same token
        # spaces, newlines, or = signs that follow
        # and then turn the remaining token as an argument
        # this way something like
        # --extra_args="--foo --bar" will parse properly
        if self.after_opt and self.partial:
            arg = self.partial.ltrim('= \t\n\r')
            # still 
            self.partial = None
            self.after_opt = False
            return Token(arg)
        
        proc = self.partial.ltrim('= \t\n\r')

        # If we get just a -- this is a SEPARATOR token 
        if proc.rtrim() == '--':
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
    def parse_args(self, ctx, args):
        tokens = ArgTokenizer(args)
        return list(self.parse_tokens(ctx, tokens))

@dataclass
class Option:
    option: str
    value: Any

class OptionParser(ArgParser):
    def __init__(self, long, short=None, has_value=False, populate=setattr):
        if long.startswith('--'):
            long = long[2:]
        if short.startswith('-'):
            short = short[1:]
        self.populate = populate
        self.long = long
        self.short = short
        self.has_value = has_value

        self.expected = [Token(TokenType.OPTIONAL_LONG, long)]
        if self.short:
            self.expected.append(Token(TokenType.OPTIONAL_SHORT, short))
   
    def parse_tokens(self, ctx, tokens):
        arg = next(tokens)
        if self.type is not None:
            value = next(tokens)
        self.populate(ctx, arg, value)

# applies parsers in parallel 
# trying one parser at a time
class ParallelParser:
    def __init__(self, *sub_parsers):
        self.sub_parsers = sub_parsers
    
    def add_parser(self, sub_parser):
        self.sub_parsers.append(sub_parser)

# Applies parsers sequentially
class SequentialParser:
    def __init__(self, *sub_parsers, loop=False):
        self.sub_parsers = sub_parsers

# -------- Utilities for parsing config options ----------

@dataclass
class ActivityConfig:
    action: Any
    configs: List[Any]

class ConfigsParser:
    def __init__(self, populate):
        pass
    
    def parse_tokens(self, cfg, tokens):
        return []