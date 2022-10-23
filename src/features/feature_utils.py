import collections
from functools import partial
import json
from pathlib import Path
import random

import nltk
from nltk.tree import Tree
import numpy as np
import pandas as pd
import scipy
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import BernoulliNB

from src.utils import data_utils, tokenizers, tree_utils


def binary_features(x):
    return {w: True for w in set(x)}


def subtrees(x, filter_=lambda t: True):
    if type(x) == str:
        return {}
    if type(x) == list:
        d = {}
        for s in x:
            d.update(subtrees(s, filter_=filter_))
        return d
    tpls, spans = tree_utils.get_tpls(x), tree_utils.get_spans(x)
    lst = filter(filter_, zip(tpls, spans))
    return binary_features(lst)


def pcfg_subtrees(x, filter_=lambda t: True):
    if type(x) == str:
        return {}
    if type(x) == list:
        d = {}
        for s in x:
            d.update(pcfg_subtrees(s, filter_=filter_))
        return d
    tpls, spans = tree_utils.get_tpls(x), tree_utils.get_spans(x, scfg=False)
    lst = filter(filter_, zip(tpls, spans))
    return binary_features(lst)


def pcfg_subtrees_by_depth(d):
    return partial(
        pcfg_subtrees, filter_=lambda t: tree_utils.tpl_depth(t[0]) <= d
    )


def tree_productions(x, filter_=lambda t: True):
    if type(x) == str:
        return {}
    productions = x.productions()
    lst = []
    for p in productions:
        lst.append((str(p.lhs()), str(p.rhs())))
    return binary_features(lst)


def trees_by_production(x, filter_=lambda t: True):
    if type(x) == str:
        return {}
    subtrees = tree_utils.get_subtrees(x)
    lst = []
    for subtree in filter(filter_, subtrees):
        if len(subtree) <= 1:
            continue
        tpl = tree_utils.root_production(subtree)
        a, b = tpl
        if type(subtree[0]) != str and type(subtree[1]) != str:
            l = tree_utils.flatten_tree(subtree[0])
            r = tree_utils.flatten_tree(subtree[1])
            s = f"({subtree.label()} {l} {r})"
        else:
            s = ""
        lst.append((f"'{a}' '{b}'", s))
    return binary_features(lst)


def combine_imdb_sents_json(preds, split, tokenizer=None):
    tokenizer = tokenizers.get_word_tokenizer("imdb_sents")
    imdb = data_utils.load_dataset("imdb", split, tokenizer)
    uid_to_p = {}
    missing = set()
    for p in preds:
        uid = p["e"]["details"]["original_uid"]
        if uid in uid_to_p:
            uid_to_p[uid]["tree"].append(p["tree"])
        elif uid not in imdb.uid_to_e:
            missing.add(uid)
        else:
            uid_to_p[uid] = {
                "uid": uid,
                "e": imdb.uid_to_e[uid],
                "tree": [p["tree"]],
            }
    print(f"{len(uid_to_p)} imdb predictions")
    return list(uid_to_p.values())


def combine_imdb_sents(data):
    uid_to_idx = collections.defaultdict(list)
    for i, uid in enumerate(data["review_id"]):
        uid_to_idx[uid].append(i)
    out = {
        "uid": [],
        "sentence": [],
        "label": [],
        "label_name": [],
        "tree": [],
        "uids": [],
    }
    for uid, idxs in uid_to_idx.items():
        if not idxs:
            continue
        out["uid"].append(uid)
        out["sentence"].append([data["sentence"][i] for i in idxs])
        out["tree"].append([data["tree"][i] for i in idxs])
        out["label"].append(data["label"][idxs[0]])
        out["label_name"].append(data["label_name"][idxs[0]])
    print(f"{len([u for u, i in uid_to_idx.items() if i])} imdb predictions")
    return out


def load_trees(output_dir, dataset=""):
    data = {k: {} for k in ("train", "dev")}
    data["dataset"] = dataset
    for k in ("train", "dev"):
        fn = Path(output_dir) / f"{k}.tsv"
        if not fn.exists():
            print(f"missing {fn}")
            continue
        data[k] = pd.read_csv(fn, delimiter="\t").to_dict(orient="list")
        data[k]["tree"] = tree_utils.strings_to_trees(data[k]["tree"])
        if dataset == "imdb":
            data[k] = combine_imdb_sents(data[k])
        data[k]["y"] = np.array(data[k]["label"])
        data[k]["uid"] = [str(uid) for uid in data[k]["uid"]]
    return data


def predictions_to_dict(preds):
    if "e" in preds[0]:
        examples = [p["e"] for p in preds]
    else:
        examples = preds
    d = {"uid": [e["uid"] for e in examples]}
    if "a" in examples[0]:
        d["sentence1"] = [e["a"]["text"] for e in examples]
        d["sentence2"] = [e["b"]["text"] for e in examples]
    else:
        d["sentence"] = [e["text"] for e in examples]
    if "original_uid" in (examples[0].get("details", {}) or {}):
        d["review_id"] = [e["details"]["original_uid"] for e in examples]
    label_names = data_utils.get_label_names(fn)
    d["label"] = [e["label"] for e in examples]
    d["label_name"] = [label_names[e["label"]] for e in examples]
    if "tree" in preds[0]:
        d["tree"] = [p["tree"] for p in preds]
    return d


def load_trees_from_predictions(output_dir, dataset=""):
    data = {k: {} for k in ("train", "dev")}
    data["dataset"] = dataset
    for k, name in (("train", f"{dataset}_train"), ("dev", f"{dataset}")):
        fn = Path(output_dir) / f"predictions.{name}.end.json"
        if not fn.exists():
            print(f"missing {fn}")
            continue
        with open(fn, "r") as f:
            preds = json.load(f)
        data[k] = predictions_to_dict(preds)
        data[k]["tree"] = tree_utils.strings_to_trees(data[k]["tree"])
        if dataset == "imdb":
            data[k] = combine_imdb_sents(data[k])
        data[k]["y"] = np.array(data[k]["label"])
        data[k]["uid"] = [str(uid) for uid in data[k]["uid"]]
    return data


def add_tokens(data, tokenizer):
    for k in ("train", "dev"):
        if "sentence1" in data[k]:
            lst_a = tokenizer.get_tokens(data[k]["sentence1"])
            lst_b = tokenizer.get_tokens(data[k]["sentence2"])
            data[k]["tokens"] = [(a, b) for a, b in zip(lst_a, lst_b)]
        else:
            data[k]["tokens"] = tokenizer.get_tokens(data[k]["sentence"])


def mutual_info(X, y, model=None):
    if model is None:
        model = BernoulliNB().fit(X, y)
    n_y = model.class_count_
    n_yz = np.stack(
        [model.feature_count_.T, n_y[np.newaxis, :] - model.feature_count_.T],
        -1,
    )
    p_yz = (n_yz + 1) / (n_yz + 1).sum((1, 2), keepdims=True)
    p_z = p_yz.sum(1, keepdims=True)
    p_y = p_yz.sum(2, keepdims=True)
    pmi = np.log(p_yz) - np.log(p_z) - np.log(p_y)
    lmi = p_yz * pmi
    mi = lmi.sum((1, 2))
    return mi, model.feature_count_.T


def add_features(data, name, func, key="tree", verbose=True, sort=True):
    if "vect" not in data:
        data["vect"] = {}
        data["idx_w"] = {}
    for split in ("train", "dev"):
        if "features" not in data[split]:
            data[split]["features"] = {}
        data[split]["features"][name] = [func(x) for x in data[split][key]]
    features = data["train"]["features"][name]
    vect = DictVectorizer(sparse=True, dtype=bool).fit(features)
    if verbose:
        print(f"{len(vect.vocabulary_)} {name} features")
    for split in ("train", "dev"):
        if "X" not in data[split]:
            data[split]["X"] = {}
        features = data[split]["features"][name]
        data[split]["X"][name] = vect.transform(features)
    X, y = data["train"]["X"][name], data["train"]["y"]
    mi, counts = mutual_info(X, y)
    order = np.argsort(-mi, axis=-1) if sort else np.arange(len(mi))
    if "mi" not in data:
        data["mi"] = {}
        data["counts"] = {}
    data["mi"][name] = mi[order]
    data["counts"][name] = counts[order]
    data["idx_w"][name] = np.array(vect.feature_names_)[order]
    for split in ("train", "dev"):
        data[split]["X"][name] = data[split]["X"][name][:, order]


def get_subtree_feature_table(data, name="Subtrees"):
    mi, counts, idx_w = (
        data["mi"][name],
        data["counts"][name],
        data["idx_w"][name],
    )
    label_names = data_utils.get_label_names(data["dataset"])
    roots = np.array([tree_utils.get_root_label(s) for s in idx_w[:, 0]])
    df = pd.DataFrame(
        {"Root": roots, "Subtree": idx_w[:, 0], "Yield": idx_w[:, 1], "MI": mi}
    )
    df["Count"] = counts.sum(-1)
    for y, label in enumerate(label_names):
        df[label] = counts[:, y]
    df["Majority label"] = np.array(
        [label_names[y] for y in np.argmax(counts, -1)]
    )
    df["% majority"] = np.max(
        (counts + 1) / (counts + 1).sum(-1, keepdims=True), -1
    )
    return df


def fmt_words(wds):
    if len(wds) == 0:
        return ""
    if type(wds) == str:
        return ", ".join(wds)
    if type(wds[0]) == str:
        return ", ".join(wds)
    if type(wds[0]) == tuple and wds[0] and type(wds[0][0]) == str:
        return ", ".join([" ".join(w) for w in wds])
    if type(wds[0]) == tuple and wds[0] and type(wds[0][0]) == tuple:
        return ", ".join([" ".join(w[0]) + "/" + " ".join(w[1]) for w in wds])
    if type(wds[0]) == np.ndarray:
        return ", ".join([w[1] for w in wds])
    print(wds, type(wds[0]))
    raise ValueError


def get_merged_feature_table(data, merge_name="merge", k=8):
    mi, counts, idx_w = (
        data["mi"][merge_name],
        data["counts"][merge_name],
        data["idx_w"][merge_name],
    )
    label_names = data_utils.get_label_names(data["dataset"])
    roots = np.array([k for k, _ in idx_w])
    # words = np.array([", ".join(wds[:k, 1]) for _, wds in idx_w])
    words = np.array([fmt_words(wds[:k]) for _, wds in idx_w])
    df = pd.DataFrame({"Root": roots, "Examples": words, "MI": mi})
    df["Count"] = counts.sum(-1)
    for y, label in enumerate(label_names):
        df[label] = counts[:, y]
    df["Majority label idx"] = np.argmax(counts, -1)
    df["Majority label"] = np.array(
        [label_names[y] for y in np.argmax(counts, -1)]
    )
    df["% majority"] = np.max(
        (counts + 1) / (counts + 1).sum(-1, keepdims=True), -1
    )
    df["Support count"] = np.max(counts, -1)
    df["Counter count"] = df["Count"] - df["Support count"]
    return df


def dense(X):
    if type(X) == scipy.sparse.csr_matrix:
        out = X.toarray()
        if len(out.shape) == 2 and out.shape[1] == 1:
            return out[:, 0]
        return out
    return X


def merge_feature_groups_by_argmax(X, y, idxs, counts, feature_keys):
    cutoff = np.argmax(counts, -1)
    c = np.zeros(counts.shape[-1])
    f = np.zeros(X.shape[0], dtype=bool)
    merged_d = collections.defaultdict(
        lambda: [[[], 0, c, f]] * counts.shape[-1]
    )
    feature_keys = np.array(feature_keys)
    for key in set(feature_keys):
        m = feature_keys == key
        merged = merged_d[key]
        for j, (group, group_mi, _, features) in enumerate(merged):
            mj = m & (j == cutoff)
            ids = idxs[mj]
            f = dense(X[:, ids].sum(-1) > 0).astype(bool).A1
            new_mi, new_counts = mutual_info(f[:, np.newaxis], y)
            merged[j] = (ids, new_mi, new_counts, f)
    return merged_d


def merge_feature_groups(X, y, idxs, mi, feature_keys):
    c = np.zeros(2)
    f = np.zeros(X.shape[0], dtype=bool)
    merged_d = collections.defaultdict(lambda: [[[], 0, c, f]])
    for i, _, key in zip(idxs, mi, feature_keys):
        merged = merged_d[key]
        scores = []
        lst = []
        f = dense(X[:, i]).astype(bool)
        mi_, c = mutual_info(f[:, np.newaxis], y)
        for j, (group, group_mi, _, features) in enumerate(merged):
            new_group = group + [i]
            new_features = features | f
            X_ = new_features[:, np.newaxis]
            new_mi, new_counts = mutual_info(X_, y)
            scores.append(np.sign(new_mi - group_mi) * new_mi)
            lst.append((new_group, new_mi, new_counts, new_features))
        if all(s < 0 for s in scores):
            merged.append([[i], mi_, c, f])
        else:
            j = np.argmax(scores)
            merged[j] = lst[j]
    return merged_d


def add_merges(
    data,
    feature_name,
    merge_name="merge",
    K=1000,
    filter_=None,
    by_argmax=True,
    by_template=False,
):
    if merge_name not in data:
        data[merge_name] = {feature_name: {}}
    key = feature_name
    if key not in data[merge_name]:
        data[merge_name][key] = {}
    X = data["train"]["X"][feature_name]
    y = data["train"]["y"]
    order = np.arange(X.shape[-1])
    if filter_ is not None:
        print("filtering")
        m = np.array([filter_(w) for w in data["idx_w"][feature_name]])
        m = order[m]
        order = order[m]
    idxs = order[:K]
    mi = data["mi"][feature_name][idxs]
    fs = data["idx_w"][feature_name][idxs][:, 0]
    if not by_template:
        fs = [tree_utils.get_root_label(s) for s in fs]
    if by_argmax:
        counts = data["counts"][feature_name][idxs]
        merged = merge_feature_groups_by_argmax(
            X=X, y=y, counts=counts, idxs=idxs, feature_keys=fs
        )
    else:
        merged = merge_feature_groups(
            X=X, y=y, mi=mi, idxs=idxs, feature_keys=fs
        )
    items = [(k, v) for k, vs in merged.items() for v in vs]
    items.sort(key=lambda it: -it[1][1])
    data[merge_name][key][K] = items
    data["idx_w"][merge_name] = [
        (k, data["idx_w"][key][v[0]]) for k, v in items
    ]
    data["mi"][merge_name] = np.concatenate([v[1] for _, v in items], 0)
    data["counts"][merge_name] = np.concatenate([v[2] for _, v in items], 0)
    data["train"]["X"][merge_name] = np.array([v[3] for _, v in items]).T
    X = data["dev"]["X"][feature_name]
    data["dev"]["X"][merge_name] = np.concatenate(
        [X[:, v[0]].sum(-1) > 0 for _, v in items], 1
    )


def add_classifier_accuracy(data, fn, name, split="dev"):
    with open(fn, "r") as f:
        preds = json.load(f)
    if type(preds) == dict:
        preds = list(preds.values())[0]
    uid_p = {p["e"]["uid"]: p for p in preds}
    if "acc" not in data[split]:
        data[split]["acc"] = {}
        data[split]["y_hat"] = {}
    data[split]["acc"][name] = np.array(
        [uid_p[uid]["acc"] for uid in data[split]["uid"]]
    )
    data[split]["y_hat"][name] = np.array(
        [uid_p[uid]["predicted"] for uid in data[split]["uid"]]
    )


def get_support_counter_accuracy(
    data, model_name, feature_name, table, split="dev"
):
    X, y = data[split]["X"][feature_name], data[split]["y"]
    d = X.shape[-1]
    if type(X) == scipy.sparse.csr_matrix:
        assert d < 10000
        X = X.todense()
    majority_y = data["counts"][feature_name].argmax(-1)
    support = X & (y[:, np.newaxis] == majority_y[np.newaxis, :])
    counter = X & (y[:, np.newaxis] != majority_y[np.newaxis, :])
    acc = data[split]["acc"][model_name]
    # support_acc = np.array([acc[support[:, i].A1].mean() for i in range(d)])
    # counter_acc = np.array([acc[counter[:, i].A1].mean() for i in range(d)])
    with np.errstate(divide="ignore", invalid="ignore"):
        support_acc = (
            (acc[:, np.newaxis] & support).sum(0) / support.sum(0)
        ).A1
        counter_acc = (
            (acc[:, np.newaxis] & counter).sum(0) / counter.sum(0)
        ).A1
    table[f"{model_name} Support"] = support_acc
    table[f"{model_name} Counter"] = counter_acc


def leaf_parts(s):
    parts = s.split("/")
    if len(parts) == 1:
        return "", parts[0]
    return "/".join(parts[:-1]), parts[-1]


def get_ngrams_from_subtree(w):
    a, b = leaf_parts(w[1])
    return (tuple(a.strip().split(" ")), tuple(b.strip().split(" ")))


def get_ngrams_from_subtrees(sts):
    return [get_ngrams_from_subtree(s) for s in sts]


def ngrams(x, n):
    if len(x) < n:
        return []
    return [tuple(x[i - n : i]) for i in range(n, len(x) + 1)]


def pair_ngrams_from_lsts_(x, lsts=[], max_n=3, max_m=3):
    a, b = x
    a_ngrams = set(w for n in range(1, max_n + 1) for w in ngrams(a, n))
    b_ngrams = set(w for n in range(1, max_m + 1) for w in ngrams(b, n))
    d = {}
    for i, lst in enumerate(lsts):
        for u, v in lst:
            has_u = u in a_ngrams
            has_v = v in b_ngrams
            if u[0] == "ϵ":
                d[i] = has_v
            elif v[0] == "ϵ":
                d[i] = has_u
            else:
                d[i] = has_u and has_v
            if d[i]:
                break
    return d


def hypothesis_ngrams_from_lsts_(x, lsts=[], max_n=3, max_m=3):
    a, b = x
    b_ngrams = set(w for n in range(1, max_m + 1) for w in ngrams(b, n))
    d = {}
    for i, lst in enumerate(lsts):
        for _, v in lst:
            d[i] = v in b_ngrams
            if d[i]:
                break
    return d


def pair_ngrams_from_lsts_max_n(lsts, max_n, max_m):
    return partial(pair_ngrams_from_lsts_, lsts=lsts, max_n=max_n, max_m=max_m)


def hypothesis_ngrams_from_lsts_max_n(lsts, max_n, max_m):
    return partial(
        hypothesis_ngrams_from_lsts_, lsts=lsts, max_n=max_n, max_m=max_m
    )


def add_equivalent_ngram_pairs(data, ngram_features):
    max_n = max(max(len(a) for a, _ in fs) for fs in ngram_features if fs)
    max_m = max(max(len(b) for _, b in fs) for fs in ngram_features if fs)
    print(max_n, max_m)
    add_features(
        data,
        "N-gram pairs",
        pair_ngrams_from_lsts_max_n(ngram_features, max_n, max_m),
        key="tokens",
        sort=False,
    )
    data["idx_w"]["N-gram pairs"] = [
        (i, ngram_features[i]) for i in data["idx_w"]["N-gram pairs"]
    ]


def add_equivalent_hypothesis_ngrams(data, ngram_features):
    max_n = max(max(len(a) for a, _ in fs) for fs in ngram_features if fs)
    max_m = max(max(len(b) for _, b in fs) for fs in ngram_features if fs)
    print(max_n, max_m)
    add_features(
        data,
        "Hypothesis n-grams",
        hypothesis_ngrams_from_lsts_max_n(ngram_features, max_n, max_m),
        key="tokens",
        sort=False,
    )
    data["idx_w"]["Hypothesis n-grams"] = [
        (i, [(("*",), v) for _, v in ngram_features[i]])
        for i in data["idx_w"]["Hypothesis n-grams"]
    ]
