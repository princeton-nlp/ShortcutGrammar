from dataclasses import asdict, dataclass, field
import json
import pandas as pd
from pathlib import Path
import pickle
import random
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.distributed import DistributedSampler


from src.utils import logging

logger = logging.get_logger(__name__)


def distributed_dataset(dataset):
    sampler = DistributedSampler(
        dataset,
        num_replicas=torch.distributed.get_world_size(),
        rank=torch.distributed.get_rank(),
        drop_last=True,
        shuffle=False,
        seed=0,
    )
    return torch.utils.data.Subset(dataset, list(sampler))


@dataclass
class Sentence:
    uid: str
    text: str
    label: Any = field(default=None)
    details: Any = field(default=None)


@dataclass
class SentencePair:
    uid: str
    a: dict
    b: dict
    label: Any = field(default=None)
    details: Any = field(default=None)


@dataclass
class ProcessedSentence:
    uid: str
    input_ids: List[int]
    label: Any = field(default=None)


def process_sentences(sentences, tokenizer, max_length=128):
    if "a" in sentences[0] and "b" in sentences[0]:
        assert "a" not in sentences[0]["a"]
        return process_sentence_pairs(sentences, tokenizer, max_length)
    logger.info(f"tokenizing {len(sentences)} sentences")
    encoded = tokenizer([s["text"] for s in sentences])
    return [
        asdict(ProcessedSentence(s["uid"], input_ids, s.get("label", None)))
        for s, input_ids in zip(sentences, encoded["input_ids"])
    ]


def process_sentence_pairs(sentence_pairs, tokenizer, max_length=128):
    a_s = process_sentences(
        [p["a"] for p in sentence_pairs], tokenizer, max_length
    )
    b_s = process_sentences(
        [p["b"] for p in sentence_pairs], tokenizer, max_length
    )
    return [
        asdict(SentencePair(p["uid"], a, b, p.get("label", None)))
        for p, a, b in zip(sentence_pairs, a_s, b_s)
    ]


def cache_fn(name, split, tokenizer, cache_dir="cache"):
    return (Path(cache_dir) / split / name).with_suffix(
        "." + tokenizer.__class__.__name__ + ".pkl"
    )


def num_labels(name):
    return 3 if "nli" in name.lower() else 2


def load_json(fn, remove_br=True):
    with open(Path(fn).with_suffix(".json"), "r") as f:
        data = json.load(f)
    examples = []
    for i, e in enumerate(data):
        if remove_br and "a" not in e and e["text"].startswith("<br"):
            e["text"] = e["text"].replace("<br />", "")
            e["text"] = e["text"].replace("<br / >", "")
        if "uid" not in e:
            e["uid"] = str(i)
        if "a" in e and "uid" not in e["a"]:
            e["a"]["uid"] = str(i)
        if "b" in e and "uid" not in e["b"]:
            e["b"]["uid"] = str(i)
        if "a" in e or e["text"].strip():
            examples.append(e)
    return examples


def label_idx(s):
    d = {
        "Entailment": 0,
        "Contradiction": 1,
        "Neutral": 2,
        "Negative": 0,
        "Positive": 1,
        "Subjective": 0,
        "Objective": 1,
        "No paraphrase": 0,
        "Paraphrase": 1,
    }
    return d.get(s, None)


def load_tsv(fn):
    data = pd.read_csv(fn.with_suffix(".tsv"), delimiter="\t")
    examples = []
    for i, row in data.iterrows():
        uid = str(row.get("uid", i))
        e = {"uid": uid}
        if "sentence1" in row:
            e["a"] = {"text": row["sentence1"], "uid": uid}
            e["b"] = {"text": row["sentence2"], "uid": uid}
        else:
            e["text"] = row["sentence"]
        if "label" in row:
            e["label"] = row["label"]
        if "label_name" in row:
            e["label_name"] = row["label_name"]
        if "review_id" in row:
            d["details"] = {"original_uid": row["review_id"]}
        examples.append(e)
    return examples


def length(f):
    if "a" in f:
        return len(f["a"]["input_ids"]) + len(f["b"]["input_ids"])
    return len(f["input_ids"])


def product_length(f):
    if "a" in f:
        return len(f["a"]["input_ids"]) * len(f["b"]["input_ids"])
    return len(f["input_ids"])


def to_single(dataset, keep="b", detail="a", cat=False):
    examples = []
    for i, e in enumerate(dataset.examples):
        e[keep]["details"] = e[detail]
        examples.append(e[keep])
        if cat:
            f = dataset.uid_to_f[e["uid"]]
            f[keep]["input_ids"] += f[detail]["input_ids"]
    processed = [f[keep] for f in dataset.processed]
    return GenericDataset(dataset.name, examples, processed)


def remove_special_tokens(dataset):
    for p in dataset.processed:
        if "a" in p:
            p["a"]["input_ids"] = p["a"]["input_ids"][1:-1]
            p["b"]["input_ids"] = p["b"]["input_ids"][1:-1]
        else:
            p["input_ids"] = p["input_ids"][1:-1]
    return dataset


def load_dataset(
    name,
    split,
    tokenizer,
    max_examples=None,
    max_length=512,
    min_length=0,
    overwrite_cache=False,
    data_seed=13,
    data_dir="data",
    cache_dir="cache",
    remove_special=False,
    convert_to_single=False,
    use_product_length=False,
    **kwargs,
):
    if name.endswith(".json"):
        fn = Path(name)
        dataset_name = str(fn.with_suffix(""))
        examples = load_json(fn)
    elif name.endswith(".tsv"):
        fn = Path(name)
        dataset_name = str(fn.with_suffix(""))
        examples = load_tsv(fn)
    else:
        fn = Path(data_dir) / name / split
        dataset_name = name
        examples = load_tsv(fn)
    cache = cache_fn(name, split, tokenizer)
    if cache_dir and cache.exists() and not overwrite_cache:
        logger.debug(f"loading processed examples from {cache}")
        with open(cache, "rb") as f:
            processed = pickle.load(f)
    else:
        processed = process_sentences(examples, tokenizer)
        if cache_dir:
            logger.info(f"caching processed examples to {cache}")
            cache.parent.mkdir(exist_ok=True, parents=True)
            with open(cache, "wb") as f:
                pickle.dump(processed, f)

    length_func = product_length if use_product_length else length
    keep = [
        (e, p)
        for e, p in zip(examples, processed)
        if (length_func(p) >= min_length) and (length_func(p) <= max_length)
    ]
    logger.debug(
        f"filtered to {len(keep)}/{len(examples)} examples "
        f"with {min_length} <= length <= {max_length}"
    )
    examples, processed = zip(*keep)

    if (max_examples or 0) > 0 and max_examples < len(examples):
        random.seed(data_seed)
        examples = random.sample(examples, max_examples)
        uids = set(e["uid"] for e in examples)
        processed = [e for e in processed if e["uid"] in uids]
        logger.info(f"picked {len(examples)} examples")

    dataset = GenericDataset(dataset_name, examples, processed)

    if remove_special:
        logger.info(f"removing special tokens ({name}/{split})")
        dataset = remove_special_tokens(dataset)

    if convert_to_single:
        logger.info(f"{name} to_single")
        dataset = to_single(dataset)

    return dataset


def collate(batch, concatenate_pair=False):
    examples, processed = zip(*batch)
    d = {"examples": examples}
    d.update(collate_(processed))
    return d


def collate_(processed):
    d = {}
    B = len(processed)
    for k in processed[0].keys():
        values = [f[k] for f in processed]
        if type(values[0]) == str:
            d[k] = values
        elif type(values[0]) in (int, float):
            d[k] = torch.tensor(values)
        elif type(values[0]) == list:
            max_len = max(len(v) for v in values)
            d[k] = torch.full((B, max_len), 0, dtype=torch.long)
            mask_k = None
            if k == "input_ids":
                mask_k = "attention_mask"
                d[mask_k] = torch.full((B, max_len), 0, dtype=torch.bool)
                d["length"] = torch.zeros((B,), dtype=torch.long)
            for i, v in enumerate(values):
                t = torch.tensor(v, dtype=torch.long)
                d[k][i, : len(t)] = t
                if k == "input_ids":
                    m = torch.ones_like(t, dtype=torch.bool)
                    d[mask_k][i, : len(m)] = m
                    d["length"][i] = len(m)
        elif type(values[0]) == dict:
            d[k] = collate_(values)
        elif type(values[0]) == torch.Tensor:
            d[k] = pad_sequence(values, batch_first=True)
        elif values[0] is not None:
            raise NotImplementedError(f"{k}: {type(values[0])}")
    return d


class GenericDataset:
    def __init__(self, name, examples, processed):
        self.name = name
        self.examples = examples
        self.processed = processed
        self.uid_to_e = {e["uid"]: e for e in examples}
        self.uid_to_f = {f["uid"]: f for f in processed}

    def __len__(self):
        return len(self.processed)

    def __getitem__(self, idx):
        f = self.processed[idx]
        e = self.uid_to_e[f["uid"]]
        return e, f

    def split(self, train_negative_examples=False):
        uids = list(self.uid_to_e.keys())
        random.shuffle(uids)
        a_uids = set(uids[: len(uids) // 2])
        a_processed = [f for f in self.processed if f["uid"] in a_uids]
        a_examples = [e for e in self.examples if e["uid"] in a_uids]
        b_processed = [f for f in self.processed if f["uid"] not in a_uids]
        b_examples = [e for e in self.examples if e["uid"] not in a_uids]
        return (
            GenericDataset(self.name, a_examples, a_processed),
            GenericDataset(self.name, b_examples, b_processed),
        )


def get_label_names(name):
    s = name.lower()
    if "nli" in s:
        return ["Entailment", "Contradiction", "Neutral"]
    if "qqp" in s:
        return ["No paraphrase", "Paraphrase"]
    if "subj" in s:
        return ["Subjective", "Objective"]
    return ["Negative", "Positive"]


def clean(token, repl="_"):
    return token.replace("Ġ", repl)


def tokens(input_ids, tokenizer):
    toks = tokenizer.convert_ids_to_tokens(input_ids)
    return [clean(t) for t in toks]


class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
