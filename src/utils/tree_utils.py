import re

import nltk
from nltk.tree import Tree
import pandas as pd

from tqdm.notebook import tqdm


def get_subtree(t, lst=None):
    for st in t:
        if isinstance(st, Tree):
            get_subtree(st, lst=lst)
    if lst is not None:
        lst.append(t)


def get_subtrees(t):
    lst = []
    get_subtree(t, lst=lst)
    return lst


def get_tpl(t, lst=None):
    out = [str(t.label())]
    for st in t:
        if isinstance(st, Tree):
            out.append(get_tpl(st, lst=lst))
        else:
            out.append("*")
    s = f"({' '.join(out)})"
    if lst is not None:
        lst.append(s)
    return s


def get_tpls(t):
    lst = []
    get_tpl(t, lst=lst)
    return lst


def tpl_depth(t):
    md = 0
    d = 0
    for c in t:
        if c == "(":
            d += 1
            md = max(d, md)
        elif c == ")":
            d -= 1
    return md


def join(l, r, scfg=False):
    if scfg:
        s = f"{' '.join(l)}/{' '.join(r)}".replace("_", "")
    else:
        s = " ".join(l)
    return s.replace("_", "")


def get_span(t, lst=None, scfg=False):
    out = []
    for st in t:
        if isinstance(st, Tree):
            out.append(get_span(st, lst=lst, scfg=scfg))
        else:
            out.append(str(st))
    l, r = [], []
    for s in out:
        if scfg:
            idx = s.rfind("/")
            a = s[:idx]
            b = s[idx + 1 :]
            l.append(a)
            r.append(b)
        else:
            l.append(s)
    s = join(l, r, scfg=scfg)
    if lst is not None:
        lst.append(s)
    return s


def get_spans(t, scfg=True):
    lst = []
    if len(t.leaves()) == 0:
        return lst
    get_span(t, lst=lst, scfg=scfg)
    if scfg:
        for i, w in enumerate(lst):
            if w.strip().endswith("/"):
                lst[i] = w.strip() + "ϵ"
            elif w.strip().startswith("/"):
                lst[i] = "ϵ" + w.strip()
    return [re.sub(r"\s+", " ", s) for s in lst]


def _child_names(tree):
    names = []
    for child in tree:
        if isinstance(child, Tree):
            names.append(nltk.tree.Nonterminal(child._label))
        else:
            names.append(child)
    return names


def get_root_label(s):
    if " " not in s:
        return ""
    return s[1 : s.index(" ")]


def root_production(t):
    return (str(t._label), str(tuple(_child_names(t))))


def fmt_production(p):
    return (str(p.lhs()), tuple(map(str, p.rhs())))


def fmt_node(node):
    if len(node) == 1:
        return f"{node.label()} -> '{node[0]}'"
    return f"{node.label()} -> {node[0].label()} {node[1].label()}"


def add_trees_to_predictions(predictions):
    skipped = 0
    for p in predictions:
        try:
            p["tree"] = Tree.fromstring(p["tree"])
        except:
            skipped += 1
    print(f"parsed {len(predictions) - skipped}/{len(predictions)} trees")


def strings_to_trees(lst):
    out = []
    skipped = 0
    for s in lst:
        try:
            t = Tree.fromstring(s)
        except:
            skipped += 1
            t = s
        out.append(t)
    print(f"parsed {len(out) - skipped}/{len(out)} trees")
    return out


def get_tree_features(
    predictions, labels=["Entailment", "Contradiction", "Neutral"]
):
    feature_rows = []
    prediction_rows = []
    for p in tqdm(predictions):
        if type(p["tree"]) == str:
            continue
        row = {
            "uid": p["e"]["uid"],
            "y": p["e"]["label"],
            "label": labels[p["e"]["label"]],
        }
        if "a" in p["e"]:
            row["a"] = p["e"]["a"]
            row["b"] = p["e"]["b"]
        prediction_rows.append(row)
        t = p["tree"]
        spans = get_spans(t)
        tpls = get_tpls(t)
        subtrees = get_subtrees(t)
        for span, tpl, subtree in zip(spans, tpls, subtrees):
            root = root_production(subtree)
            row = {
                "uid": p["e"]["uid"],
                "y": p["e"]["label"],
                "label": labels[p["e"]["label"]],
                "span": span,
                "tpl": tpl,
                "subtree": str(subtree),
                "root": root[0],
                "root_production": root,
            }
            feature_rows.append(row)
    return feature_rows, prediction_rows
