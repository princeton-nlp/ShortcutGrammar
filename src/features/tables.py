import argparse
import collections
import copy
import itertools
import json
import logging
from pathlib import Path
import pandas as pd
import random

from src.utils import data_utils, feature_utils, metrics, tree_utils

from tqdm.notebook import tqdm
import numpy as np
import scipy


def parse_args(use_args=None, namespace=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="")
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--cmd", type=str, default="")


def fmt_stderr(v, std=None):
    if type(v) == list:
        std = np.std(v)
        v = np.mean(v)
    v = np.round(100 * v, 1)
    std = np.round(100 * std, 1)
    return f"{v}\\textpm{std}"


def fmt_val(v):
    v = np.round(100 * v, 1)
    return str(v)


def fmt_dataset(d):
    lookup = {
        "hans": "HANS",
        "SNLI_cad": "SNLI CAD",
        "breaking_nli": "Break",
        "SNLI_short": "SNLI",
        "QQP16": "QQP",
    }
    return lookup.get(d, d)


def upsampling_table_compact_s(paths, datasets, print_csv=False):
    reports = []
    for (name, path) in paths:
        fn = path / "metrics.end.json"
        if not fn.exists():
            print(f"missing {fn}")
            continue
        with open(fn, "r") as f:
            report = json.load(f)
            reports.append((name, report))
    rows = []
    for name, report in reports:
        row = [name]
        for dataset in datasets:
            row.append(fmt_val(report[dataset]["acc"]))
        rows.append(row)
    header = ["Method"] + list(map(fmt_dataset, datasets))
    feature_utils.markdown_table(
        header, [[s.replace("\\textpm", "/") for s in r] for r in rows]
    )
    print(" & ".join(header) + "\\\\")
    print("\\midrule")
    for row in rows:
        print(" & ".join(list(map(str, row))) + "\\\\")
    if print_csv:
        feature_utils.print_csv(
            header, [[s.replace("\\textpm", "/") for s in r] for r in rows]
        )


def upsampling_table_compact(
    paths,
    datasets,
    print_latex=True,
    print_csv=False,
    print_features=False,
    splits=None,
    K=3,
):
    reports = []
    for (name, path) in paths:
        fn = path / "metrics.avg.json"
        if not fn.exists():
            print(f"missing {fn}")
            continue
        with open(fn, "r") as f:
            report = json.load(f)
            report["key"] = path.name
            reports.append((name, report))
    rows = []
    for name, report in reports:
        key = report["key"]
        s = name
        row = [s]
        if (
            print_features
            and ("merge" in key or "nt_group" in key)
            and "features" in splits[key]
        ):
            s = fmt_features(splits[key]["features"][:K]) or s
            if "nt_group" in key:
                s = f"Group {s}"
            row.append(s)
        elif print_features:
            row.append("")
        for dataset in datasets:
            row.append(fmt_stderr(report[dataset]["acc_lst"]))
        rows.append(row)
    header = (
        ["Method"]
        + (["Examples"] if print_features else [])
        + list(map(fmt_dataset, datasets))
    )
    feature_utils.markdown_table(header, rows)
    if print_latex:
        feature_utils.print_latex(header, rows)
    if print_csv:
        feature_utils.print_csv(header, rows)


def upsampling_table(ds, datasets, counts_from=None):
    rows = []
    for (p, d) in ds:
        name = p.parent.name.upper()
        setting = p.name
        if "L" in setting:
            a, b = setting.split("L")
            K = int(a[1:])
            L = b
            setting = f"K={K}, $\\lambda$={L}"
        row = [
            name,
            setting,
        ]
        if "supporting_acc_lst" in d[counts_from]:
            row += [
                int(d[counts_from]["supporting_count"]),
                fmt_stderr(d[counts_from]["supporting_acc_lst"]),
                int(d[counts_from]["counter_count"]),
                fmt_stderr(d[counts_from]["counter_acc_lst"]),
            ]
        else:
            row += ["", "", "", ""]
        for dataset in datasets:
            row.append(fmt_stderr(d[dataset]["acc_lst"]))
        rows.append(row)
    header = [
        "Method",
        "Setting",
        "Support",
        "Support Acc.",
        "Counter",
        "Counter Acc.",
    ] + list(map(fmt_dataset, datasets))
    print(" & ".join(header) + "\\\\")
    print("\\midrule")
    for row in rows:
        print(" & ".join(list(map(str, row))) + "\\\\")


def get_edit_preds(path, name, dataset):
    edit_preds = {}
    for k in ("original", "edits"):
        fn = Path(path) / f"predictions.{dataset}_edits_{name}_{k}.end.json"
        with open(fn, "r") as f:
            edit_preds[k] = json.load(f)
    for p, p_ in zip(edit_preds["original"], edit_preds["edits"]):
        p_["original"] = p
    return edit_preds


def fmt_edit(ed):
    if type(ed["w"][0]) in (tuple, list):
        w = "/".join([" ".join(w) for w in ed["w"]])
        w_ = "/".join([" ".join(w) for w in ed["w_"]])
    else:
        w = " ".join(ed["w"])
        w_ = " ".join(ed["w_"])
    return f"{w}\\textarrow {w_}"


def change_lmi(d):
    keys = sorted(d.keys(), key=lambda k: d[k].sum())
    n_xy = np.stack([d[k] for k in keys], 0)
    p_xy = (n_xy + 1) / (n_xy + 1).sum()
    p_x = p_xy.sum(-1, keepdims=True)
    p_y = p_xy.sum(0, keepdims=True)
    pmi = np.log(p_xy / (p_x * p_y))
    lmi = p_xy * pmi
    return {k: l for k, l in zip(keys, lmi)}


def group_edits_by_tpl(
    path=None,
    name=None,
    dataset=None,
    preds=None,
    N=10,
    K=3,
    print_=True,
    return_changed=False,
):
    if preds is None:
        preds = get_edit_preds(path, name, dataset)["edits"]
    tpls = collections.defaultdict(list)
    tpl_wds = collections.defaultdict(collections.Counter)
    tpl_changed = collections.defaultdict(lambda: np.zeros(2))
    for p in preds:
        ed = p["e"]["details"]["edits"][0]
        tpls[ed["tpl"]].append(p)
        change = int(p["predicted"] != p["original"]["predicted"])
        tpl_wds[ed["tpl"]][fmt_edit(ed)] += change
        tpl_changed[ed["tpl"]][change] += 1
    # by_pct = sorted(tpl_changed.keys(), key=lambda tpl: -tpl_changed[tpl][1])
    # tpl_pct = {k: v[1] / v.sum() for k, v in tpl_changed.items()}
    tpl_pct = change_lmi(tpl_changed)
    by_pct = sorted(tpl_changed.keys(), key=lambda tpl: -tpl_pct[tpl][1])
    rows = []
    for tpl in by_pct[:N]:
        pct = round(100 * tpl_changed[tpl][1] / tpl_changed[tpl].sum(), 2)
        row = [tpl]
        row.append(", ".join([str(w) for w, _ in tpl_wds[tpl].most_common(K)]))
        row += [str(len(tpls[tpl])), pct]
        if print_:
            print(" & ".join(list(map(str, row))) + "\\\\")
        rows.append(row)
    if return_changed:
        return {
            k: [p for p in ps if p["predicted"] != p["original"]["predicted"]]
            for k, ps in tpls.items()
        }
    if not print_:
        return rows


def group_edits_by_edit(path=None, name=None, dataset=None, preds=None, N=10):
    if preds is None:
        preds = get_edit_preds(path, name, dataset)["edits"]
    edits = collections.defaultdict(lambda: np.zeros(2))
    for p in preds:
        ed = p["e"]["details"]["edits"][0]
        k = (ed["tpl"], fmt_edit(ed))
        change = int(p["predicted"] != p["original"]["predicted"])
        edits[k][change] += 1
    by_pct = sorted(edits.keys(), key=lambda ed: -edits[ed][1])
    for ed in by_pct[:N]:
        pct = round(100 * edits[ed][1] / edits[ed].sum(), 2)
        row = [str(ed), str(edits[ed].sum()), str(edits[ed][1])]
        print(" & ".join(list(map(str, row))) + "\\\\")


def get_top_edits(path=None, name=None, dataset=None, preds=None, K=3):
    if preds is None:
        preds = get_edit_preds(path, name, dataset)["edits"]
    edits = collections.defaultdict(lambda: np.zeros(2))
    for p in preds:
        ed = p["e"]["details"]["edits"][0]
        k = (ed["tpl"], fmt_edit(ed))
        change = int(p["predicted"] != p["original"]["predicted"])
        edits[k][change] += 1
    by_pct = sorted(edits.keys(), key=lambda ed: -edits[ed][1])
    # edit_pct = {k: v[1] / v.sum() for k, v in edits.items()}
    # by_pct = sorted(edits.keys(), key=lambda ed: -edit_pct[ed])
    return by_pct[:K]


def contrast_row(path, name, dataset, N=3, K=3):
    edit_preds = get_edit_preds(path, name, dataset)
    wrong = [
        p_
        for p, p_ in zip(edit_preds["original"], edit_preds["edits"])
        if p["acc"] and not p_["acc"]
    ]
    prev_correct = [
        p_
        for p, p_ in zip(edit_preds["original"], edit_preds["edits"])
        if p["acc"]
    ]
    changed = [
        p_
        for p, p_ in zip(edit_preds["original"], edit_preds["edits"])
        if p["predicted"] != p_["predicted"]
    ]
    pct = round(100 * len(changed) / len(edit_preds["edits"]), 2)
    # wpct = round(100 * len(wrong) / len(prev_correct), 2)
    if K > 0:
        # edits = get_top_edits(preds=edit_preds["edits"], K=K)
        edit_rows = group_edits_by_tpl(
            preds=edit_preds["edits"], N=N, K=3, print_=False
        )
        for i, row in enumerate(edit_rows):
            s = "& " + " & ".join(list(map(str, row))) + "\\\\"
            s = s.replace("##", "{\#\#}")
            if i == 0:
                s = fmt_dataset(dataset) + " " + s
            print(s)
    r = " & ".join(["", "", "", str(len(edit_preds["edits"])), str(pct)])
    print(r + "\\\\")
    print("\\midrule")


def get_splits(splits):
    d = {}
    for name, fn in splits:
        with open(fn, "r") as f:
            d[name] = json.load(f)
    return d


def get_split_row_single(output_dir, dataset, splits):
    fn = Path(output_dir) / f"predictions.{dataset}.end.json"
    if not fn.exists():
        return [], {}
    with open(fn, "r") as f:
        preds = json.load(f)
    report, _ = metrics.score_classifier_predictions(preds)
    for split_name, split_fn in splits:
        data_utils.add_group_info_to_preds(preds, split_fn, name=split_name)
        metrics.add_group_reports(preds, report, name=split_name)
        report["key"] = Path(output_dir).parent.name
    return preds, report


def fmt(s):
    return s.replace("_", " ").capitalize()


def fmtd(d):
    return str(np.round(100 * d, 1))


def get_split_row(
    output_dir, dataset, splits, seeds=[7, 9, 13, 17, 19], avg=True
):
    if not avg:
        return get_split_row_single(output_dir, dataset, splits)
    lst = []
    for s in seeds:
        p, r = get_split_row_single(Path(output_dir) / f"s{s}", dataset, splits)
        if p and r:
            lst.append((p, r))
    if len(lst) == 0:
        return {}
    report = metrics.average_dicts([r for _, r in lst], force_lst=True)
    if "key" in report and type(report["key"]) == list:
        report["key"] = report["key"][0]
    return report


def get_split_reports(model_dirs, dataset, splits, seeds=[7, 9, 13, 17, 19]):
    reports = []
    for name, output_dir in model_dirs:
        report = get_split_row(output_dir, dataset, splits, seeds=seeds)
        if report:
            reports.append((name, report))
            print(name, report["acc"])
        else:
            print(name, "missing")
    return reports


def fmt_features(lst):
    if len(lst) and type(lst[0]) != str:
        # lhs = lst[0][0]
        # rhs = [f[1] for f in lst]
        # return f"{lhs}: " + ", ".join(rhs)
        return ", ".join([f[1] for f in lst])
    return ", ".join(lst)


def split_table_all_splits(
    reports,
    splits,
    print_features=False,
    K=8,
    print_latex=True,
    print_csv=False,
):
    header = ["Name"]
    keys = list(splits.keys())
    for split in keys:
        d = splits[split]
        if print_features and "features" in d:
            s = fmt_features(d["features"][:K]) or split
            header += [f"{split} ({s}) S", "C"]
        else:
            header += [split + " S", "C"]
    rows = []
    for name, report in reports:
        row = [name]
        for key in keys:
            row += [
                fmt_stderr(report[f"{key}_supporting_acc_lst"]),
                fmt_stderr(report[f"{key}_counter_acc_lst"]),
            ]
        rows.append(row)
    feature_utils.markdown_table(header, rows)
    if print_latex:
        feature_utils.print_latex(header, rows)
    if print_csv:
        feature_utils.print_csv(header, rows)


def split_table(
    original,
    reports,
    splits=None,
    print_features=False,
    K=8,
    print_latex=True,
    print_csv=False,
):
    header = ["Name"]
    if print_features:
        header += ["Examples"]
    header += [
        "Supporting",
        "BERT S",
        "Drift S",
        "Counter",
        "BERT C",
        "Drift C",
    ]
    rows = []
    for name, report in reports:
        key = report["key"]
        s = name
        row = [s]
        if (
            print_features
            and ("merge" in key or "nt_group" in key)
            and "features" in splits[key]
        ):
            s = fmt_features(splits[key]["features"][:K]) or s
            if "nt_group" in key:
                s = f"Group {s}"
            row.append(s)
        elif print_features:
            row.append("")
        row += [
            str(int(report[f"{key}_supporting_count"])),
            fmt_stderr(original[f"{key}_supporting_acc_lst"]),
            fmt_stderr(report[f"{key}_supporting_acc_lst"]),
            str(int(report[f"{key}_counter_count"])),
            fmt_stderr(original[f"{key}_counter_acc_lst"]),
            fmt_stderr(report[f"{key}_counter_acc_lst"]),
        ]
        rows.append(row)
    feature_utils.markdown_table(header, rows)
    if print_latex:
        feature_utils.print_latex(header, rows)
    if print_csv:
        feature_utils.print_csv(header, rows)


def guess_labels(data):
    name = data["dataset"].lower()
    if "nli" in name:
        return ["Entailment", "Contradiction", "Neutral"]
    if "sst2" in name or "sst-2" in name:
        return ["Negative", "Postive"]
    return ["No", "Yes"]


def grouped_merged_feature_table_compact(
    data,
    items,
    feature_name,
    K=10,
    print_md=True,
    print_csv=True,
    print_latex=True,
    labels=None,
    ft_names=["BERT"],
    sort_by_label=False,
    row_limit=None,
    single_key=False,
    hard_counter=False,
):
    k_to_g = collections.defaultdict(list)
    k_to_mi = collections.defaultdict(float)
    for i, (k, (g, mi, f)) in enumerate(items):
        k_to_g[k].append((k, (g, mi, f)))
        k_to_mi[k] += mi
    groups = sorted(k_to_g.items(), key=lambda it: -k_to_mi[it[0]])

    if labels is None:
        labels = guess_labels(data)
    rows = []
    for _, item_group in groups[:K]:
        for k, (g, m, f) in item_group:
            if single_key:
                ws = [
                    str(w)
                    for _, w in data["idx_w"][feature_name][g][:row_limit]
                ]
                row = [", ".join(ws)]
            else:
                ws = [
                    str(w)
                    for _, w in data["idx_w"][feature_name][g][:row_limit]
                ]
                row = [k, ", ".join(ws)]
            counts = feature_utils.label_counts_(
                data["train"]["y"][f], len(labels)
            ).astype(int)
            acc = np.mean(data["train"]["y"][f] == np.argmax(counts))
            dev_f = dense(data["dev"]["X"][feature_name][:, g]).sum(-1) > 0
            dev_acc = np.mean(data["dev"]["y"][dev_f] == np.argmax(counts))

            # row.append(np.round(100 * acc, 1))
            # row.append(np.round(100 * dev_acc, 1))

            row += list(map(str, counts))

            y = data["dev"]["y"]
            y_hat = np.argmax(counts)
            support = dev_f & (y == y_hat)
            if hard_counter:
                counter = dev_f & (y == np.argmin(counts))
            else:
                counter = dev_f & (y != y_hat)

            for ft in ft_names:
                if np.any(support):
                    ft_sup = np.mean(data["dev"][f"{ft}_acc"][support])
                    row.append(np.round(100 * ft_sup, 1))
                else:
                    row.append("-")
                if np.any(counter):
                    ft_counter = np.mean(data["dev"][f"{ft}_acc"][counter])
                    row.append(np.round(100 * ft_counter, 1))
                else:
                    row.append("-")

            rows.append(row)
    header = ([] if single_key else ["Key"]) + ["Productions"] + labels
    for ft in ft_names:
        header += [f"{ft} acc (support)", f"{ft} acc (counter)"]

    if sort_by_label:
        rows = sorted(
            rows,
            key=lambda r: np.argmax(list(map(int, r[2 : 2 + len(labels)]))),
        )

    feature_utils.markdown_table(header, rows)
    if print_csv:
        feature_utils.print_csv(
            header, [[str(s).replace("\\textpm", "/") for s in r] for r in rows]
        )
    if print_latex:
        feature_utils.print_latex(header, rows)


def dense(X):
    if type(X) == scipy.sparse.csr_matrix:
        out = X.toarray()
        if len(out.shape) == 2 and out.shape[1] == 1:
            return out[:, 0]
        return out
    return X


def merged_feature_table(
    data,
    items,
    feature_name,
    K=10,
    print_csv=True,
    print_latex=True,
    labels=None,
    hard_counter=False,
):
    if labels is None:
        labels = guess_labels(data)
    rows = []
    for k, (g, m, f) in items[:K]:
        ws = [str(w) for _, w in data["idx_w"][feature_name][g]]
        row = [k, ", ".join(ws), str(np.round(m, 5))]
        counts = feature_utils.label_counts_(
            data["train"]["y"][f], len(labels)
        ).astype(int)
        acc = np.mean(data["train"]["y"][f] == np.argmax(counts))
        # dev_f = data["dev"]["X"][feature_name][:, g].sum(-1) > 0
        dev_f = dense(data["dev"]["X"][feature_name][:, g]).sum(-1) > 0
        dev_acc = np.mean(data["dev"]["y"][dev_f] == np.argmax(counts))

        row.append(np.round(100 * acc, 1))
        row.append(np.round(100 * dev_acc, 1))

        row += list(map(str, counts))

        y = data["dev"]["y"]
        y_hat = np.argmax(counts)
        support = dev_f & (y == y_hat)
        counter = dev_f & (y != y_hat)
        bert_sup = np.mean(data["dev"]["BERT_acc"][support])
        bert_counter = np.mean(data["dev"]["BERT_acc"][counter])

        row.append(np.round(100 * bert_sup, 1))
        row.append(np.round(100 * bert_counter, 1))

        rows.append(row)
    header = (
        ["key", "words", "MI", "Acc. given z (train)", "Acc. given z (dev)"]
        + labels
        + ["BERT acc (support)", "BERT acc (counter)"]
    )

    feature_utils.markdown_table(header, rows)
    if print_csv:
        feature_utils.print_csv(
            header, [[str(s).replace("\\textpm", "/") for s in r] for r in rows]
        )
    if print_latex:
        feature_utils.print_latex(header, rows)


def get_group_ft_acc_by_label(data, g, feature_name, ft_names=["BERT"]):
    f = data["train"]["X"][feature_name][:, g].sum(-1) > 0
    f = f.getA()[:, 0]
    num_labels = len(guess_labels(data))
    counts = feature_utils.label_counts_(
        data["train"]["y"][f], num_labels
    ).astype(int)
    y = data["dev"]["y"]
    dev_f = data["dev"]["X"][feature_name][:, g].sum(-1) > 0
    dev_f = dev_f.getA()[:, 0]
    dev_acc = np.mean(data["dev"]["y"][dev_f] == y_hat)
    fs = [dev_f & (y == y_) for y_ in range(num_labels)]
    support = dev_f & (y == y_hat)
    counter = dev_f & (y != y_hat)

    row = []
    for ft in ft_names:
        if np.any(support):
            ft_sup = np.mean(data["dev"][f"{ft}_acc"][support])
            row.append(np.round(100 * ft_sup, 1))
        else:
            row.append("-")
        if np.any(counter):
            ft_counter = np.mean(data["dev"][f"{ft}_acc"][counter])
            row.append(np.round(100 * ft_counter, 1))
        else:
            row.append("-")
    return row


def get_group_support_counter(
    data, g, feature_name, split="dev", ft_names=["BERT"], y_hat=None
):
    f = data["train"]["X"][feature_name][:, g].sum(-1) > 0
    f = f.getA()[:, 0]
    counts = feature_utils.label_counts_(
        data["train"]["y"][f], len(guess_labels(data))
    ).astype(int)
    y = data[split]["y"]
    if y_hat is None:
        y_hat = np.argmax(counts)
    dev_f = data[split]["X"][feature_name][:, g].sum(-1) > 0
    dev_f = dev_f.getA()[:, 0]
    dev_acc = np.mean(data[split]["y"][dev_f] == y_hat)
    support = dev_f & (y == y_hat)
    counter = dev_f & (y != y_hat)
    return support, counter


def get_group_ft_acc(data, g, feature_name, ft_names=["BERT"], y_hat=None):
    f = data["train"]["X"][feature_name][:, g].sum(-1) > 0
    f = f.getA()[:, 0]
    counts = feature_utils.label_counts_(
        data["train"]["y"][f], len(guess_labels(data))
    ).astype(int)
    y = data["dev"]["y"]
    if y_hat is None:
        y_hat = np.argmax(counts)
    dev_f = data["dev"]["X"][feature_name][:, g].sum(-1) > 0
    dev_f = dev_f.getA()[:, 0]
    dev_acc = np.mean(data["dev"]["y"][dev_f] == y_hat)
    support = dev_f & (y == y_hat)
    counter = dev_f & (y != y_hat)

    row = []
    for ft in ft_names:
        if np.any(support):
            ft_sup = np.mean(data["dev"][f"{ft}_acc"][support])
            row.append(np.round(100 * ft_sup, 1))
        else:
            row.append("-")
        if np.any(counter):
            ft_counter = np.mean(data["dev"][f"{ft}_acc"][counter])
            row.append(np.round(100 * ft_counter, 1))
        else:
            row.append("-")
    return row


def merged_feature_table_compact(
    data,
    items,
    feature_name,
    K=10,
    print_md=True,
    print_csv=False,
    print_latex=False,
    labels=None,
    ft_names=["BERT"],
    sort_by_label=True,
    row_limit=8,
    single_key=False,
    hard_counter=False,
):
    if labels is None:
        labels = guess_labels(data)
    rows = []
    for k, (g, m, f) in items[:K]:
        if single_key:
            ws = [
                " ".join(w) for w in data["idx_w"][feature_name][g][:row_limit]
            ]
            row = [", ".join(ws)]
        else:
            ws = [str(w) for _, w in data["idx_w"][feature_name][g][:row_limit]]
            row = [k, ", ".join(ws)]

        counts = feature_utils.label_counts_(
            data["train"]["y"][f], len(labels)
        ).astype(int)
        acc = np.mean(data["train"]["y"][f] == np.argmax(counts))
        dev_f = dense(data["dev"]["X"][feature_name][:, g]).sum(-1) > 0
        dev_acc = np.mean(data["dev"]["y"][dev_f] == np.argmax(counts))

        # row.append(np.round(100 * acc, 1))
        # row.append(np.round(100 * dev_acc, 1))

        row += list(map(str, counts))

        y = data["dev"]["y"]
        y_hat = np.argmax(counts)
        support = dev_f & (y == y_hat)
        if hard_counter:
            counter = dev_f & (y == np.argmin(counts))
        else:
            counter = dev_f & (y != y_hat)

        for ft in ft_names:
            if np.any(support):
                ft_sup = np.mean(data["dev"][f"{ft}_acc"][support])
                row.append(np.round(100 * ft_sup, 1))
            else:
                row.append("-")
            if np.any(counter):
                ft_counter = np.mean(data["dev"][f"{ft}_acc"][counter])
                row.append(np.round(100 * ft_counter, 1))
            else:
                row.append("-")

        rows.append(row)
    header = ([] if single_key else ["Key"]) + ["Productions"] + labels
    for ft in ft_names:
        header += [f"{ft} acc (support)", f"{ft} acc (counter)"]

    if sort_by_label:
        rows = sorted(
            rows,
            key=lambda r: np.argmax(list(map(int, r[2 : 2 + len(labels)]))),
        )

    if print_md:
        feature_utils.markdown_table(header, rows, bold_left=not single_key)
    if print_csv:
        feature_utils.print_csv(
            header, [[str(s).replace("\\textpm", "/") for s in r] for r in rows]
        )
    if print_latex:
        feature_utils.print_latex(header, rows)


def merged_feature_table_compact_v3(
    data,
    items,
    feature_name,
    K=10,
    descriptions=None,
    print_csv=False,
    print_latex=False,
    labels=None,
    ft_names=[],
    sort_by_label=True,
    row_limit=8,
    single_key=False,
    hard_counter=False,
):
    if labels is None:
        labels = guess_labels(data)
    rows = []
    if descriptions is None:
        descriptions = [""] * K
    for (k, (g, m, f)), desc in zip(items[:K], descriptions):
        row = ["", desc]
        if desc == "":
            row[-1] = fmt_features(data["idx_w"][feature_name][g[:3]])
        counts = feature_utils.label_counts_(
            data["train"]["y"][f], len(labels)
        ).astype(int)
        count = counts.sum()
        p_y = np.mean(data["train"]["y"][f] == np.argmax(counts))
        row[0] = np.argmax(counts)
        dev_f = dense(data["dev"]["X"][feature_name][:, g]).sum(-1) > 0
        dev_acc = np.mean(data["dev"]["y"][dev_f] == np.argmax(counts))

        # row.append(np.round(100 * acc, 1))
        # row.append(np.round(100 * dev_acc, 1))
        # row += [count, np.round(100 * p_y, 1)]

        # row += list(map(str, counts))

        y = data["dev"]["y"]
        y_hat = np.argmax(counts)
        support = dev_f & (y == y_hat)
        if hard_counter:
            counter = dev_f & (y == np.argmin(counts))
        else:
            counter = dev_f & (y != y_hat)
        row += [int(support.sum()), int(counter.sum())]
        # row[-1] += f" ({int(support.sum())}/{int(counter.sum())})"

        for ft in ft_names:
            if np.any(support):
                ft_sup = np.mean(data["dev"][f"{ft}_acc"][support])
                row.append(np.round(100 * ft_sup, 1))
            else:
                row.append("-")
            if np.any(counter):
                ft_counter = np.mean(data["dev"][f"{ft}_acc"][counter])
                row.append(np.round(100 * ft_counter, 1))
            else:
                row.append("-")

        rows.append(row)
    header = ["Class", "Description", "\# (support)", "\# (counter)"]
    # header = ["Class", "Description (\#S, \#C)"]
    for ft in ft_names:
        header += [f"{ft} acc (support)", f"{ft} acc (counter)"]

    if sort_by_label:
        rows = sorted(rows, key=lambda r: r[0])
    rows = [[""] + r[1:] for r in rows]

    feature_utils.markdown_table(header, rows, bold_left=not single_key)
    if print_csv:
        feature_utils.print_csv(
            header, [[str(s).replace("\\textpm", "/") for s in r] for r in rows]
        )
    if print_latex:
        feature_utils.print_latex(header, rows)


def merged_feature_table_compact_v2(
    data,
    items,
    feature_name,
    K=10,
    descriptions=None,
    print_csv=False,
    print_latex=False,
    labels=None,
    ft_names=[],
    sort_by_label=True,
    row_limit=8,
    single_key=False,
    hard_counter=False,
):
    if labels is None:
        labels = guess_labels(data)
    rows = []
    if descriptions is None:
        descriptions = [""] * K
    for (k, (g, m, f)), desc in zip(items[:K], descriptions):
        if single_key:
            ws = [
                " ".join(w) for w in data["idx_w"][feature_name][g][:row_limit]
            ]
            row = [", ".join(ws)]
        else:
            ws = [str(w) for _, w in data["idx_w"][feature_name][g][:row_limit]]
            row = ["", k, desc, ", ".join(ws)]

        counts = feature_utils.label_counts_(
            data["train"]["y"][f], len(labels)
        ).astype(int)
        count = counts.sum()
        p_y = np.mean(data["train"]["y"][f] == np.argmax(counts))
        if hard_counter:
            p_y = np.mean(data["train"]["y"][f] != np.argmin(counts))
        row[0] = np.argmax(counts)
        dev_f = dense(data["dev"]["X"][feature_name][:, g]).sum(-1) > 0
        dev_acc = np.mean(data["dev"]["y"][dev_f] == np.argmax(counts))

        # row.append(np.round(100 * acc, 1))
        # row.append(np.round(100 * dev_acc, 1))
        row += [count, np.round(100 * p_y, 1)]

        # row += list(map(str, counts))

        y = data["dev"]["y"]
        y_hat = np.argmax(counts)
        support = dev_f & (y == y_hat)
        if hard_counter:
            support = dev_f & (y != np.argmin(counts))
            counter = dev_f & (y == np.argmin(counts))
        else:
            counter = dev_f & (y != y_hat)

        for ft in ft_names:
            if np.any(support):
                ft_sup = np.mean(data["dev"][f"{ft}_acc"][support])
                row.append(np.round(100 * ft_sup, 1))
            else:
                row.append("-")
            if np.any(counter):
                ft_counter = np.mean(data["dev"][f"{ft}_acc"][counter])
                row.append(np.round(100 * ft_counter, 1))
            else:
                row.append("-")

        rows.append(row)
    header = ([] if single_key else ["Key"]) + ["Productions"] + labels
    header = ["Class", "Root", "Description", "Examples", "Count", "\%"]
    for ft in ft_names:
        header += [f"{ft} acc (support)", f"{ft} acc (counter)"]

    if sort_by_label:
        rows = sorted(rows, key=lambda r: r[0])
    rows = [[""] + r[1:] for r in rows]

    feature_utils.markdown_table(header, rows, bold_left=not single_key)
    if print_csv:
        feature_utils.print_csv(
            header, [[str(s).replace("\\textpm", "/") for s in r] for r in rows]
        )
    if print_latex:
        feature_utils.print_latex(header, rows)


def equivalent_group_table_single(
    data,
    item_idx,
    src_feature,
    tgt_features,
    K=1000,
    num_rows=10,
    print_latex=True,
    print_csv=True,
    include_mi=False,
    use_y_hat=False,
):
    src_key, src_feature_name = src_feature
    idx_w = data["idx_w"][src_feature_name]
    counts = data["counts"][src_feature_name]
    rows = []
    # header = ["Feature", "MI", "Count", "$p(y\mid z)$", "Drop"]
    # for t, _ in tgt_features:
    #     header += [f"{t} {k}" for k in ["MI", "Count", "$p(y\mid z)$", "Drop"]]
    header = ["Feature"]
    for k in (["MI"] if include_mi else []) + ["Count", "$p(y\mid z)$", "Drop"]:
        for t, _ in [src_feature] + tgt_features:
            header.append(f"{t} {k}")
    k, (g, m, f) = item = data["merges"][src_key][K][item_idx]
    src_g = g
    rows = []
    for i in range(min(len(src_g), num_rows)):
        j = src_g[i]
        # row = [" ".join(idx_w[j])]
        row = [idx_w[j][1]]
        lsts = []
        group_y_hat = None
        for key, feature_name in [src_feature] + tgt_features:
            k, (g, m, f) = item = data["merges"][key][K][item_idx]
            j = g[i]
            c = data["counts"][feature_name][j]
            p = (c + 1) / (c + 1).sum()
            y_hat = np.argmax(c)
            if group_y_hat is None:
                group_y_hat = y_hat
            mi = data["mi"][feature_name][j]
            fs, fc = get_group_ft_acc(
                data,
                [j],
                feature_name,
                y_hat=group_y_hat if use_y_hat else None,
            )
            # row += [
            #     np.round(mi, 4),
            #     int(c.sum()),
            #     np.round(p[y_hat], 3),
            #     f"{fs}/{fc}",
            # ]
            lsts.append(
                ([np.round(mi, 4)] if include_mi else [])
                + [
                    int(c.sum()),
                    np.round(p[y_hat], 3),
                    f"{fs}/{fc}",
                ]
            )
        for i in range(len(lsts[0])):
            row += [l[i] for l in lsts]
        rows.append(row)
    feature_utils.markdown_table(header, rows)
    if print_csv:
        feature_utils.print_csv(header, rows)
    if print_latex:
        feature_utils.print_latex(header, rows)


def equivalent_group_table_pd(
    data,
    src_feature,
    tgt_features,
    K=1000,
    num_rows=10,
    print_latex=True,
    print_csv=True,
    include_mi=False,
    use_y_hat=False,
    sort_by_y=True,
    nt_filter=None,
):
    src_key, src_feature_name = src_feature
    idx_w = data["idx_w"][src_feature_name]
    counts = data["counts"][src_feature_name]
    rows = []
    src_items = data["merges"][src_key][K]
    tgt_items_lst = [data["merges"][tgt_key][K] for tgt_key, _ in tgt_features]
    groups_added = 0
    ys = []
    for i, item_ in enumerate(src_items):
        tgt_items = [lst[i] for lst in tgt_items_lst]
        # row = [" ".join(idx_w[j])]
        k, (g, m, f) = item_
        wds = idx_w[g[:3]]
        root = tree_utils.get_root_label(wds[0, 0])
        if nt_filter is not None and nt_filter(root):
            continue
        groups_added += 1
        wds = fmt_features(wds)
        group_y_hat = None
        for (key, feature_name), item in zip(
            [src_feature] + tgt_features, [item_] + tgt_items
        ):
            k, (g, m, f) = item
            if type(f) == np.matrix:
                f = f.getA()[:, 0]
            c = feature_utils.label_counts_(
                data["train"]["y"][f], len(guess_labels(data))
            ).astype(int)
            p = (c + 1) / (c + 1).sum()
            y_hat = np.argmax(c)
            mi = m
            if group_y_hat is None:
                group_y_hat = y_hat
                ys.append(group_y_hat)
            fs, fc = get_group_ft_acc(
                data, g, feature_name, y_hat=group_y_hat if use_y_hat else None
            )
            row = {
                "root": root,
                "words": wds,
                "y_hat": group_y_hat,
                "key": key,
                "mi": mi,
                "count": c.sum(),
                "p_y": p[group_y_hat],
                "support_acc": fs,
                "counter_acc": fc,
            }
            rows.append(row)
        if num_rows and groups_added >= num_rows:
            break
    return pd.DataFrame(rows)


def equivalent_group_table_mult(
    data,
    src_feature,
    tgt_features,
    K=1000,
    num_rows=10,
    print_md=True,
    print_latex=True,
    print_csv=True,
    include_mi=False,
    use_y_hat=False,
    sort_by_y=True,
    nt_filter=None,
    descriptions=None,
    description_d=None,
    merge_key="am_merges",
):
    src_key, src_feature_name, _ = src_feature
    idx_w = data["idx_w"][src_feature_name]
    counts = data["counts"][src_feature_name]
    rows = []
    pd_rows = []
    # header = ["Feature", "MI", "Count", "$p(y\mid z)$", "Drop"]
    # for t, _ in tgt_features:
    #     header += [f"{t} {k}" for k in ["MI", "Count", "$p(y\mid z)$", "Drop"]]
    header = ["Label", "Examples"]
    # for k in ["MI", "Count", "$p(y\mid z)$", "Drop"]:
    for k in (["MI"] if include_mi else []) + [
        "Count",
        "$p(y\mid z)$",
        "Support",
        "Counter",
        "Drop",
    ]:
        for _, _, t in [src_feature] + tgt_features:
            header.append(f"{t} {k}")
    rows = []
    ys = []
    src_items = data[merge_key][src_key][K]
    tgt_items_lst = [
        data[merge_key][tgt_key][K] for tgt_key, _, _ in tgt_features
    ]
    labels = guess_labels(data)

    for i, item_ in enumerate(src_items):
        tgt_items = [lst[i] for lst in tgt_items_lst]
        # row = [" ".join(idx_w[j])]
        k, (g, m, f) = item_
        wds = idx_w[g[:3]]
        root = tree_utils.get_root_label(wds[0, 0])
        if nt_filter is not None and nt_filter(root):
            continue
        if descriptions is not None:
            row = ["", descriptions[i]]
            pd_row = {"Root": "", "Description": descriptions[i]}
        else:
            row = [root, fmt_features(wds)]
            pd_row = {"Root": root, "Description": fmt_features(wds)}
        lsts = []
        group_y_hat = None
        for (key, feature_name, feature_label), item in zip(
            [src_feature] + tgt_features, [item_] + tgt_items
        ):
            k, (g, m, f) = item
            if type(f) == np.matrix:
                f = f.getA()[:, 0]
            c = feature_utils.label_counts_(
                data["train"]["y"][f], len(guess_labels(data))
            ).astype(int)
            p = (c + 1) / (c + 1).sum()
            y_hat = np.argmax(c)
            mi = m
            if group_y_hat is None:
                group_y_hat = y_hat
                ys.append(group_y_hat)
            fs, fc = get_group_ft_acc(
                data, g, feature_name, y_hat=group_y_hat if use_y_hat else None
            )
            # row += [
            #     np.round(mi, 4),
            #     int(c.sum()),
            #     np.round(p[y_hat], 3),
            #     f"{fs}/{fc}",
            # ]
            pd_row_ = copy.deepcopy(pd_row)
            yk = group_y_hat if use_y_hat else y_hat
            if description_d and pd_row["Description"] in description_d:
                pd_row_["Shortcut"] = description_d[pd_row["Description"]]
            elif description_d:
                pd_row_["Shortcut"] = ""
                print(pd_row["Description"])
            pd_row_["y"] = yk
            pd_row_["Label"] = labels[yk]
            pd_row_["Feature"] = feature_label
            pd_row_["Count"] = int(c.sum())
            pd_row_["p(y | z)"] = np.round(
                100 * p[group_y_hat if use_y_hat else y_hat], 1
            )
            pd_row_["Support"] = fs
            pd_row_["Counter"] = fc
            pd_row_["Drop"] = fs - fc
            pd_rows.append(pd_row_)
            lsts.append(
                ([np.round(mi, 4)] if include_mi else [])
                + [
                    int(c.sum()),
                    np.round(100 * p[group_y_hat if use_y_hat else y_hat], 1),
                    str(round(fs, 1)),
                    str(round(fc, 1)),
                    str(round(fs - fc, 1))
                    # f"{fs}/{fc}",
                ]
            )
        for i in range(len(lsts[0])):
            row += [l[i] for l in lsts]
        rows.append(row)
        if len(rows) >= num_rows:
            break
    if sort_by_y:
        order = np.argsort(ys)
        rows = [rows[i] for i in order]
        pd_rows = sorted(pd_rows, key=lambda r: r["y"])
    if print_md:
        feature_utils.markdown_table(header, rows)
    if print_csv:
        feature_utils.print_csv(header, rows)
    if print_latex:
        feature_utils.print_latex(header, rows)
    return pd.DataFrame(pd_rows)


def add_edit_acc(preds, edit_preds):
    uid_p = {p["e"]["uid"]: p for p in preds}
    out = {}
    missing = 0
    for p in edit_preds:
        uid = p["e"]["details"]["original_uid"]
        g = p["e"]["group"]
        if uid not in uid_p:
            missing += 1
            continue
        p_ = uid_p[uid]
        if (uid, g) not in out:
            out[(uid, g)] = copy.deepcopy(p_)
        p_ = out[(uid, g)]
        if "edits" not in p_:
            p_["edits"] = []
        p_["edits"].append(p)
    if missing:
        print(f"missing {missing}")
    return list(out.items())


def edit_error_rate(p1, p2, tgt):
    seed = 13
    with open(p1, "r") as f:
        preds = json.load(f)
    with open(p2, "r") as f:
        edit_preds = json.load(f)
    edits = add_edit_acc(preds, edit_preds)
    out = []
    avg = []
    for _, p in edits:
        orig = p["predicted"]
        if orig == tgt:
            continue
        out.append(any(d["predicted"] == tgt for d in p["edits"]))
        avg.append(np.mean([int(d["predicted"] == tgt) for d in p["edits"]]))
    return len(out), np.mean(out), np.mean(avg)
