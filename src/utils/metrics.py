import numpy as np


def score_predictions(predictions):
    report = {"loss": np.mean([p["loss"] for p in predictions])}
    return report, predictions


def score_classifier_predictions(predictions):
    for p in predictions:
        p["acc"] = int(p["predicted"] == p["e"]["label"])
    report = {}
    for k in ("acc", "loss"):
        report[k] = np.mean([p[k] for p in predictions])
    return report, predictions


def average_dicts(dicts, short=False, force_lst=False):
    if len(dicts) == 1 and not force_lst:
        return dicts[0]
    avg = {}
    for k, v in dicts[0].items():
        vs = [d[k] for d in dicts if k in d]
        if type(v) == dict:
            avg[k] = average_dicts(vs)
        elif type(v) in (int, float, np.float64):
            avg[k] = np.mean(vs)
            avg[f"{k}_std"] = np.std(vs)
            if not short:
                avg[f"{k}_lst"] = vs
        elif type(vs[0]) == list and len(vs[0]) == 1:
            avg[k] = [v[0] for v in vs]
        else:
            avg[k] = vs
    return avg
