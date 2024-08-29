import sentencepiece
import os
import numpy as np
import itertools
import io

from foundry.core.dataclasses import dataclass, field


class TokenizationConfig:
    vocab_size: int 

class Tokenizer:
    def __init__(self, model=None):
        self.model = model
        self.processor = sentencepiece.SentencePieceProcessor(model_proto=model)
        self.vocab = list([self.processor.id_to_piece(id) for id in range(self.processor.get_piece_size())])
        self.vocab_size = self.processor.get_piece_size()

    def save_model(self, model_file, vocab_file=None):
        with open(model_file, "wb") as f:
            f.write(self.model)
        if vocab_file is not None:
            with open(vocab_file, "w") as f:
                f.write("\n".join([f"{v} - {i}" for i,v in enumerate(self.vocab)]))

    @staticmethod
    def load_model(model_file):
        with open(model_file, "rb") as f:
            model = f.read()
        return Tokenizer(model)
    
    def encode(self, text):
        return self.processor.encode(text)
    
    def encode_to_file(self, iterator, file, bos=False, eos=False):
        bos = self.processor.bos_id() if bos else None
        eos = self.processor.eos_id() if eos else None
        off = 0
        size = 2048
        buffer = np.zeros((size,), dtype=np.int16)
        for s in itertools.batched(iterator, 2048):
            s = list(s)
            encoded = self.processor.encode_as_ids(s, num_threads=os.cpu_count())
            total_len = sum(len(e) for e in encoded)
            # reallocate the buffer
            if off + total_len > size:
                buffer[:off].tofile(file)
                off = 0
            if total_len > size:
                size = 2*(total_len//2048 + 1)*2048
                buffer = np.zeros((size,), dtype=np.int16)
            # add to the buffer
            for e in encoded:
                l = len(e)
                buffer[off:off+l] = np.array(e, dtype=np.int16)
                off = off + l
        buffer[:off].tofile(file)

    @staticmethod
    def train(input_iterator, *, 
                vocab_size,
                input_format="text",
                character_coverage=1.0,
                num_threads=os.cpu_count(),
                split_digits=True,
                user_defined_symbols=None,
                unk_surface=r" \342\201\207 ",
                model_type="bpe",
                normalization_rule_name="identity"):
        model = io.BytesIO()
        sentencepiece.SentencePieceTrainer.train(
            sentence_iterator=input_iterator,
            user_defined_symbols=user_defined_symbols,
            model_writer=model,
            vocab_size=vocab_size,
            input_format=input_format,
            character_coverage=character_coverage,
            num_threads=num_threads,
            split_digits=split_digits,
            unk_surface=unk_surface,
            model_type=model_type,
            normalization_rule_name=normalization_rule_name
        )
        return Tokenizer(model.getvalue())

# Try and chunk the text
# to keep the text within a certain character limit
def chunk_text(lines, line_separator, char_limit):
    first = True
    while lines:
        l = 0
        for i in range(len(lines)):
            l = l + len(lines[i])
            if l > char_limit:
                break
        sentence, lines = lines[:i+1], lines[i+1:]
        story = line_separator.join(sentence)
        if not first:
            story = line_separator + story
        yield story
        first = False

def filter_ascii(sentences):
    for s in sentences:
        if s.isascii():
            yield s

def iterate_raw(data_path, entry_separator, line_symbol, char_limit):
    with open(data_path, "r") as f:
        lines = []
        while True:
            data = f.readline()
            if not data:
                break
            data = data.strip()
            if not data:
                continue
            if data == entry_separator:
                yield from chunk_text(lines, line_symbol, char_limit)
                lines = []
            else:
                lines.append(data)
        if lines:
            yield from chunk_text(lines, line_symbol, char_limit)