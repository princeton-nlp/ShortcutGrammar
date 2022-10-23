from pathlib import Path
import random

import numpy as np
import tokenizers
from tokenizers import (
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)

from src.utils import data_utils, logging


logger = logging.get_logger(__name__)


def word_tokenizer_name(dataset, vocab_size, min_frequency):
    return f"{dataset}WordTokenizer_v{vocab_size}_f{min_frequency}"


class WordTokenizer:
    def __init__(
        self, tokenizer, dataset=None, vocab_size=None, min_frequency=None
    ):
        self.tokenizer = tokenizer
        special = ["[PAD]", "[CLS]", "[SEP]", "[UNK]"]
        self.pad_token, self.cls_token, self.sep_token, self.unk_token = special
        (
            self.pad_token_id,
            self.cls_token_id,
            self.sep_token_id,
            self.unk_token_id,
        ) = map(tokenizer.token_to_id, special)
        self.bos_token_id, self.bos_token = self.cls_token_id, self.cls_token
        self.eos_token_id, self.eos_token = self.sep_token_id, self.sep_token
        self.vocab = tokenizer.get_vocab()
        self.idx_w = {i: w for w, i in self.vocab.items()}
        if vocab_size and (min_frequency is not None):
            self.__class__.__name__ = word_tokenizer_name(
                dataset, vocab_size, min_frequency
            )

    def __call__(self, sents):
        encoded = self.tokenizer.encode_batch(sents)
        return {"input_ids": [e.ids for e in encoded]}

    def __len__(self):
        return self.tokenizer.get_vocab_size()

    def convert_ids_to_tokens(self, idxs, **kwargs):
        t = self.tokenizer
        if idxs is not None and len(idxs) > 0 and type(idxs[0]) == list:
            return [self.convert_ids_to_tokens(ids, **kwargs) for ids in idxs]
        return [t.id_to_token(idx) for idx in idxs]

    def add_special_tokens(self, special):
        return self.tokenizer.add_special_tokens(special)

    def get_tokens(self, sents):
        encoded = self.tokenizer.encode_batch(
            [s if type(s) == str else "" for s in sents]
        )
        return [e.tokens[1:-1] for e in encoded]

    def save(self, fn):
        self.tokenizer.save(str(fn))


def get_preprocessor():
    normalizer = normalizers.BertNormalizer(lowercase=True)
    pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    model = models.WordLevel(unk_token="[UNK]")
    word_tokenizer = tokenizers.Tokenizer(model)
    word_tokenizer.normalizer = normalizer
    word_tokenizer.pre_tokenizer = pre_tokenizer
    special = ["[PAD]", "[CLS]", "[SEP]", "[UNK]"]
    word_tokenizer.add_special_tokens(special)
    return word_tokenizer


def build_word_tokenizer(
    name, vocab_size=20000, min_frequency=3, data_dir="data"
):
    examples = data_utils.load_tsv(Path(data_dir) / name / "train")
    logger.info(f"building word tokenizer from {len(examples)} {name} examples")
    normalizer = normalizers.BertNormalizer(lowercase=True)
    pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.BertPreTokenizer()])
    processor = processors.BertProcessing(cls=("[CLS]", 1), sep=("[SEP]", 2))
    model = models.WordLevel(unk_token="[UNK]")
    if "a" in examples[0]:
        sents = [e["a"]["text"] for e in examples] + [
            e["b"]["text"] for e in examples
        ]
    else:
        sents = [e["text"] for e in examples]
    word_tokenizer = tokenizers.Tokenizer(model)
    word_tokenizer.normalizer = normalizer
    word_tokenizer.pre_tokenizer = pre_tokenizer
    word_tokenizer.post_processor = processor
    special = ["[PAD]", "[CLS]", "[SEP]", "[UNK]"]
    if "tacred" in name:
        special += data_utils.tacred_special_tokens()
    word_tokenizer.add_special_tokens(special)
    trainer = trainers.WordLevelTrainer(
        special_tokens=special,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        show_progress=True,
    )
    random.seed(13)
    word_tokenizer.train_from_iterator(sents, trainer=trainer)
    return word_tokenizer


def get_word_tokenizer(
    name,
    vocab_size=20000,
    min_frequency=3,
    data_dir="data",
    overwrite_cache=False,
):
    tokenizer_name = word_tokenizer_name(name, vocab_size, min_frequency)
    cache_fn = (Path(data_dir) / name / tokenizer_name).with_suffix(".json")
    if not cache_fn.exists() or overwrite_cache:
        word_tokenizer = build_word_tokenizer(
            name,
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            data_dir=data_dir,
        )
        logger.info(f"caching tokenizer to {cache_fn}")
        word_tokenizer.save(str(cache_fn))
    logger.debug(f"loading tokenizer from {cache_fn}")
    word_tokenizer = Tokenizer.from_file(str(cache_fn))
    return WordTokenizer(
        word_tokenizer,
        dataset=name,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
    )


def get_tokenizer(
    train_on="",
    vocab_size=20000,
    min_frequency=3,
    data_dir="data",
    overwrite_cache=False,
    load_tokenizer_from="",
    **kwargs,
):
    if load_tokenizer_from and load_tokenizer_from.endswith("json"):
        return WordTokenizer(Tokenizer.from_file(str(load_tokenizer_from)))
    elif load_tokenizer_from:
        return WordTokenizer(Tokenizer.from_pretrained(load_tokenizer_from))
    else:
        return get_word_tokenizer(
            train_on,
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            data_dir=data_dir,
            overwrite_cache=overwrite_cache,
        )
