import sentencepiece
import os

from stanza.dataclasses import dataclass, field


class TokenizationConfig:
    vocab_size: int = field(pytree_node=False)

class Tokenizer:
    def __init__(self, model=None):
        pass


    @staticmethod
    def train(input_file, output_file, *, 
                vocab_size,
                input_format="text",
                character_coverage=1.0,
                num_threads=os.cpu_count(),
                split_digits=True,
                unk_surface=r" \342\201\207 ",
                model_type="bpe",
                normalization_rule_name="identity"):
        pass