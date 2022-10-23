import argparse
from argparse import Namespace
import copy
import json
import math
from pathlib import Path
import pickle
from pprint import pprint
import random
from scipy.special import softmax

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from tqdm.notebook import tqdm as notebook_tqdm
import matplotlib.pyplot as plt

from allennlp.common.util import sanitize
from allennlp.interpret.saliency_interpreters import (
    IntegratedGradient,
    SimpleGradient,
    SmoothGradient,
)
from allennlp.predictors import Predictor, TextClassifierPredictor

from allennlp.data.dataset_readers.text_classification_json import (
    TextClassificationJsonReader,
)
from allennlp.data.tokenizers.pretrained_transformer_tokenizer import (
    PretrainedTransformerTokenizer,
)
from allennlp.data import Instance, Token, Vocabulary
from allennlp.data.data_loaders import SimpleDataLoader
from allennlp.data.fields import TextField, LabelField, Field, TensorField
from allennlp.data.fields.text_field import TextFieldTensors
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.models import BasicClassifier, Model
from allennlp.modules.token_embedders import TokenEmbedder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder

from src.models import classifier
from src.utils import (
    data_utils,
    logging,
    metrics,
    model_utils,
    grammar_utils,
    feature_utils,
)
from scripts import model_scripts


logger = logging.get_logger(__name__)


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def parse_args(use_args=None, namespace=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--cmd", type=str, default="")
    parser.add_argument("--load_from", type=str, default="")
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--tokenizer", type=str, default="bert-base-uncased")
    parser.add_argument("--notebook", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--interpreter", type=str, default="simple_mean")
    parser.add_argument("--max_steps", type=int, default=None)

    args = parser.parse_args(args=use_args)
    if args.output_dir and not Path(args.output_dir).exists():
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"output dir: {args.output_dir}")
    logger.info(f"output dir: {args.output_dir}")

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    return args


class FakeEmbedder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, **inputs):
        return None


class FModel(Model):
    def __init__(self, vocab, model):
        super().__init__(vocab)
        self.model = model

    def forward(self, tokens=None, label=None, **kwargs):
        return self.model(
            input_ids=kwargs.get("input_ids", tokens["tokens"]["tokens"]),
            attention_mask=kwargs["attention_mask"],
            label=label,
        )


class FPredictor(Predictor):
    def __init__(self, model, reader):
        super().__init__(model, reader)
        self.fake_embedder = FakeEmbedder()

    def get_interpretable_text_field_embedder(self):
        return self.fake_embedder

    def _json_to_instance(self, d):
        instance = self._dataset_reader.text_to_instance(d["text"], d["label"])
        #         if "input_ids" in d:
        #             instance.fields["tokens"] = d["input_ids"]
        l = len(d.get("input_ids", instance.fields["tokens"]))
        if "attention_mask" in d:
            instance.add_field(
                "attention_mask", TensorField(d["attention_mask"], dtype=bool)
            )
        else:
            instance.add_field(
                "attention_mask", TensorField(np.ones(l), dtype=bool)
            )
        if "input_ids" in d:
            instance.add_field(
                "input_ids", TensorField(d["input_ids"], dtype=torch.long)
            )
        instance.add_field(
            "label", LabelField(int(d["label"]), skip_indexing=True)
        )
        return instance

    def json_to_labeled_instances(self, inputs):
        instances = self._batch_json_to_instances(inputs)
        outputs = self.predict_batch_instance(instances)
        new_instances = self.predictions_to_labeled_instances(
            instances, outputs
        )
        return new_instances

    def predictions_to_labeled_instances(self, instances, outputs):
        out = []
        for instance, output in zip(instances, outputs):
            new_instance = instance.duplicate()
            # label = np.argmax(output["log_probs"])
            # new_instance.add_field("label", LabelField(int(label), skip_indexing=True))
            out.append(new_instance)
        return out


class FSimpleGradient(SimpleGradient):
    def _aggregate_token_embeddings(self, embeddings_list, token_offsets):
        return [e.cpu().numpy() for e in embeddings_list]


class FIntegratedGradient(IntegratedGradient):
    def _aggregate_token_embeddings(self, embeddings_list, token_offsets):
        return [e.cpu().numpy() for e in embeddings_list]


class FSmoothGradient(SmoothGradient):
    def _aggregate_token_embeddings(self, embeddings_list, token_offsets):
        return [e.cpu().numpy() for e in embeddings_list]


class FSimpleGradientMean(SimpleGradient):
    def _aggregate_token_embeddings(self, embeddings_list, token_offsets):
        return [e.cpu().numpy() for e in embeddings_list]

    def saliency_interpret_from_json(self, inputs):
        labeled_instances = self.predictor.json_to_labeled_instances(inputs)
        instances_with_grads = dict()
        for idx, instance in enumerate(labeled_instances):
            # List of embedding inputs, used for multiplying gradient by the input for normalization
            embeddings_list = []
            token_offsets = []

            # Hook used for saving embeddings
            handles = self._register_hooks(embeddings_list, token_offsets)
            try:
                grads = self.predictor.get_gradients([instance])[0]
            finally:
                for handle in handles:
                    handle.remove()

            # Gradients come back in the reverse order that they were sent into the network
            embeddings_list.reverse()
            token_offsets.reverse()
            embeddings_list = self._aggregate_token_embeddings(
                embeddings_list, token_offsets
            )

            for key, grad in grads.items():
                # Get number at the end of every gradient key (they look like grad_input_[int],
                # we're getting this [int] part and subtracting 1 for zero-based indexing).
                # This is then used as an index into the reversed input array to match up the
                # gradient and its respective embedding.
                input_idx = int(key[-1]) - 1
                # The [0] here is undo-ing the batching that happens in get_gradients.
                print(grad[0].shape, embeddings_list[input_idx][0].shape)
                emb_grad = np.mean(
                    grad[0] * embeddings_list[input_idx][0], axis=1
                )
                grads[key] = emb_grad

            instances_with_grads["instance_" + str(idx + 1)] = grads
        return sanitize(instances_with_grads)


class FIntegratedGradientMean(IntegratedGradient):
    def _aggregate_token_embeddings(self, embeddings_list, token_offsets):
        return [e.cpu().numpy() for e in embeddings_list]

    def saliency_interpret_from_json(self, inputs):
        # Convert inputs to labeled instances
        labeled_instances = self.predictor.json_to_labeled_instances(inputs)

        instances_with_grads = dict()
        for idx, instance in enumerate(labeled_instances):
            # Run integrated gradients
            grads = self._integrate_gradients(instance)

            # Normalize results
            for key, grad in grads.items():
                # The [0] here is undo-ing the batching that happens in get_gradients.
                embedding_grad = np.mean(grad[0], axis=1)
                grads[key] = embedding_grad

            instances_with_grads["instance_" + str(idx + 1)] = grads

        return sanitize(instances_with_grads)


class FSimpleGradientSigned(SimpleGradient):
    def _aggregate_token_embeddings(self, embeddings_list, token_offsets):
        return [e.cpu().numpy() for e in embeddings_list]

    def saliency_interpret_from_json(self, inputs):
        labeled_instances = self.predictor.json_to_labeled_instances(inputs)
        instances_with_grads = dict()
        for idx, instance in enumerate(labeled_instances):
            # List of embedding inputs, used for multiplying gradient by the input for normalization
            embeddings_list = []
            token_offsets = []

            # Hook used for saving embeddings
            handles = self._register_hooks(embeddings_list, token_offsets)
            try:
                grads = self.predictor.get_gradients([instance])[0]
            finally:
                for handle in handles:
                    handle.remove()

            # Gradients come back in the reverse order that they were sent into the network
            embeddings_list.reverse()
            token_offsets.reverse()
            embeddings_list = self._aggregate_token_embeddings(
                embeddings_list, token_offsets
            )

            for key, grad in grads.items():
                # Get number at the end of every gradient key (they look like grad_input_[int],
                # we're getting this [int] part and subtracting 1 for zero-based indexing).
                # This is then used as an index into the reversed input array to match up the
                # gradient and its respective embedding.
                input_idx = int(key[-1]) - 1
                # The [0] here is undo-ing the batching that happens in get_gradients.
                #                 emb_grad = np.mean(grad[0] * embeddings_list[input_idx][0], axis=1)
                #                 grads[key] = emb_grad
                emb_grad = np.sum(
                    grad[0] * embeddings_list[input_idx][0], axis=1
                )
                norm = np.linalg.norm(emb_grad, ord=1)
                normalized_grad = [e / norm for e in emb_grad]
                grads[key] = normalized_grad

            instances_with_grads["instance_" + str(idx + 1)] = grads
        return sanitize(instances_with_grads)


class FIntegratedGradientSigned(IntegratedGradient):
    def _aggregate_token_embeddings(self, embeddings_list, token_offsets):
        return [e.cpu().numpy() for e in embeddings_list]

    def saliency_interpret_from_json(self, inputs):
        # Convert inputs to labeled instances
        labeled_instances = self.predictor.json_to_labeled_instances(inputs)

        instances_with_grads = dict()
        for idx, instance in enumerate(labeled_instances):
            # Run integrated gradients
            grads = self._integrate_gradients(instance)

            # Normalize results
            for key, grad in grads.items():
                # The [0] here is undo-ing the batching that happens in get_gradients.
                #                 embedding_grad = np.mean(grad[0], axis=1)
                #                 grads[key] = embedding_grad
                embedding_grad = numpy.sum(grad[0], axis=1)
                norm = numpy.linalg.norm(embedding_grad, ord=1)
                normalized_grad = [e / norm for e in embedding_grad]
                grads[key] = normalized_grad

            instances_with_grads["instance_" + str(idx + 1)] = grads

        return sanitize(instances_with_grads)


class FSmoothGradientSigned(SmoothGradient):
    def _aggregate_token_embeddings(self, embeddings_list, token_offsets):
        return [e.cpu().numpy() for e in embeddings_list]

    def saliency_interpret_from_json(self, inputs):
        # Convert inputs to labeled instances
        labeled_instances = self.predictor.json_to_labeled_instances(inputs)

        instances_with_grads = dict()
        for idx, instance in enumerate(labeled_instances):
            # Run smoothgrad
            grads = self._smooth_grads(instance)

            # Normalize results
            for key, grad in grads.items():
                # The [0] here is undo-ing the batching that happens in get_gradients.
                embedding_grad = numpy.sum(grad[0], axis=1)
                norm = numpy.linalg.norm(embedding_grad, ord=1)
                normalized_grad = [e / norm for e in embedding_grad]
                grads[key] = normalized_grad

            instances_with_grads["instance_" + str(idx + 1)] = grads

        return sanitize(instances_with_grads)


def split_tokens(preds):
    for p in preds:
        idx = np.where(p["tokens"] == "[CLS]")[0][1]
        p["a_tokens"] = p["tokens"][:idx]
        p["a_grads"] = p["grads"][:idx]
        p["b_tokens"] = p["tokens"][idx:]
        p["b_grads"] = p["grads"][idx:]


def get_interpreter(name):
    d = {
        "simple_l2": FSimpleGradient,
        "integrated_l2": FIntegratedGradient,
        "simple_mean": FSimpleGradientMean,
        "integrated_mean": FIntegratedGradientMean,
    }
    return d[name]


def pairs_to_json(batch):
    out = []
    model_inputs = model_utils.concatenate_pair(batch)
    for i, e_ in enumerate(batch["examples"]):
        e = copy.deepcopy(e_)
        e["text"] = e["a"]["text"] + "\t" + e["b"]["text"]
        e["input_ids"] = model_inputs["input_ids"][i]
        e["attention_mask"] = model_inputs["attention_mask"][i]
        out.append(e)
    return out


def get_model(args):
    with open(Path(args.load_from) / "args.json", "r") as f:
        ft_arg_d = json.load(f)
    ft_args = dotdict(**ft_arg_d)
    ft_args.load_from = args.load_from
    ft_args.output_dir = args.output_dir
    print(ft_args)
    ft_trainer = classifier.ClassifierTrainer()
    tokenizer, ft_model = ft_trainer.initialize(ft_args)
    ft_model = ft_trainer.load_from(ft_args, ft_args.load_from, ft_model)
    ft_model = ft_model.eval().to(torch.device(args.device))
    return ft_model, tokenizer


def load_datasets(args, tokenizer):
    train_dataset = data_utils.load_dataset(args.dataset, "train", tokenizer)
    dev_dataset = data_utils.load_dataset(args.dataset, "dev", tokenizer)
    return train_dataset, dev_dataset


def get_predictor(args, model, tokenizer):
    atokenizer = PretrainedTransformerTokenizer(model_name=args.tokenizer)
    reader = TextClassificationJsonReader(
        tokenizer=atokenizer, skip_label_indexing=True
    )
    vocab = Vocabulary.from_pretrained_transformer(args.tokenizer)
    fmodel = FModel(vocab, model)
    return FPredictor(fmodel, reader)


def get_batches(lst, batch_size):
    batches = []
    for i in range(0, len(lst), batch_size):
        batches.append(lst[i : i + batch_size])
    return batches


def get_labels(name):
    if "nli" in name.lower():
        return ["Entailment", "Contradiction", "Neutral"]
    if "qqp" in name.lower():
        return ["No paraphrase", "Paraphrase"]
    return ["Negative", "Positive"]


def get_num_labels(name):
    if "nli" in name.lower():
        return 3
    return 2


def run_one(args, dataset, interpreter, idx_w, max_steps=None):
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, collate_fn=data_utils.collate
    )
    uids = []
    tokens = []
    grads = []
    num_labels = get_num_labels(args.dataset)
    tq = notebook_tqdm if args.notebook else tqdm
    for step, batch in enumerate(tq(dataloader, total=len(dataloader))):
        inputs = pairs_to_json(batch) if "a" in batch else batch["examples"]
        bs = len(inputs)
        lst = []
        for y in range(num_labels):
            for e in inputs:
                e["label"] = y
            out = interpreter.saliency_interpret_from_json(inputs)
            lst.append(
                [out[f"instance_{i+1}"]["grad_input_1"] for i in range(bs)]
            )
        gs = [np.stack(ls, -1) for ls in zip(*lst)]
        ts = [idx_w[e["input_ids"].detach().cpu().numpy()] for e in inputs]
        ms = [e["attention_mask"].cpu().numpy() for e in inputs]
        grads += [g[m] for g, m in zip(gs, ms)]
        tokens += [t[m] for t, m in zip(ts, ms)]
        uids += [e["uid"] for e in batch["examples"]]
        if max_steps and step >= max_steps:
            break
    return uids, tokens, grads


def run_one_sent(args, model, tokenizer, s, name="integrated_mean"):
    predictor = get_predictor(args, model, tokenizer)
    _idx_w = np.array(
        tokenizer.convert_ids_to_tokens(np.arange(len(tokenizer)))
    )
    interpreter = get_interpreter(name)(predictor)
    d = model_scripts.to_dataset(tokenizer, s)
    _, tokens, grads = run_one(args, d, interpreter, _idx_w)
    return tokens[0], grads[0]


def get_predictions(args, dataset, predictor, max_steps=None):
    predictions = []
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, collate_fn=data_utils.collate
    )
    tq = notebook_tqdm if args.notebook else tqdm
    for step, batch in enumerate(tq(dataloader, total=len(dataloader))):
        inputs = pairs_to_json(batch) if "a" in batch else batch["examples"]
        preds = predictor.predict_batch_json(inputs)
        for p, e in zip(preds, batch["examples"]):
            p["predicted"] = np.argmax(p["log_probs"])
            p["acc"] = int(p["predicted"] == e["label"])
            p["e"] = e
        predictions += preds
        if max_steps and step >= max_steps:
            break
    return predictions


def run_attribution(args):
    model, tokenizer = get_model(args)
    dataset = data_utils.load_dataset(args.dataset, "dev", tokenizer)
    idx_w = np.array(tokenizer.convert_ids_to_tokens(np.arange(len(tokenizer))))
    predictor = get_predictor(args, model, tokenizer)
    cls = get_interpreter(args.interpreter)
    interpreter = cls(predictor)
    logger.info(f"getting predictions")
    preds = get_predictions(args, dataset, predictor, max_steps=args.max_steps)
    logger.info(f"acc: {np.mean([p['acc'] for p in preds])}")
    logger.info(f"getting {args.interpreter} gradients")
    _, tokens, grads = run_one(
        args, dataset, interpreter, idx_w, max_steps=args.max_steps
    )
    for p, t, g in zip(preds, tokens, grads):
        p["tokens"] = t
        p["grads"] = g
    fn = Path(args.output_dir) / f"{args.interpreter}.pkl"
    logger.info(f"writing to {fn}")
    with open(fn, "wb") as f:
        pickle.dump(preds, f)


def plot_heatmap_single_(
    values,
    tokens,
    center=True,
    title="",
    label="",
    do_softmax=False,
    alpha=1.0,
    **kwargs,
):
    fig, ax = plt.subplots(figsize=(12, 6))
    vmin = vmax = None
    if do_softmax:
        values = (values + alpha) / (values + alpha).sum(-1, keepdims=True)
    if center:
        low = np.min(values)
        hi = np.max(values)
        m = max(abs(low), abs(hi))
        vmin, vmax = -m, m
    im = ax.imshow(values[:, np.newaxis], vmin=vmin, vmax=vmax, **kwargs)
    ax.set_xticks([])
    ax.set_yticks(np.arange(len(tokens)))
    ax.set_yticklabels(tokens)
    if title:
        ax.set_title(title)
    plt.colorbar(im, label=label or "score")
    plt.tight_layout()
    plt.show()


def plot_heatmap_(
    values,
    tokens,
    center=False,
    title="",
    label="",
    labels=get_labels("SNLI"),
    do_softmax=False,
    alpha=1.0,
    savefig="",
    **kwargs,
):
    fig, ax = plt.subplots(figsize=(6, 10))
    vmin = vmax = None
    if do_softmax:
        values = (values + alpha) / (values + alpha).sum(-1, keepdims=True)
    if center:
        low = np.min(values)
        hi = np.max(values)
        m = max(abs(low), abs(hi))
        vmin, vmax = -m, m
    im = ax.imshow(values, vmin=vmin, vmax=vmax, **kwargs)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticks(np.arange(len(tokens)))
    ax.set_yticklabels(tokens)
    if title:
        ax.set_title(title)
    plt.colorbar(im, label=label or "Count")
    plt.tight_layout()
    if savefig:
        plt.savefig(savefig, bbox_inches="tight")
    plt.show()


def plot_heatmap_single(
    grads_d,
    tokens,
    labels=get_labels("SNLI"),
    names=None,
    center=True,
    **kwargs,
):
    if names is None:
        names = list(grads_d.keys())
    fig, axes = plt.subplots(1, len(names), figsize=(12, 6))
    if len(names) == 1:
        axes = [axes]
    vmin = vmax = None
    if center:
        low = min(np.min(grads_d[name]) for name in names)
        hi = max(np.max(grads_d[name]) for name in names)
        m = max(abs(low), abs(hi))
        vmin, vmax = -m, m
    for name, ax in zip(names, axes):
        grads = -grads_d[name]
        im = ax.imshow(grads[:, np.newaxis], vmin=vmin, vmax=vmax, **kwargs)
        ax.set_xticks([])
        ax.set_yticks(np.arange(len(tokens)))
        ax.set_yticklabels(tokens)
        ax.set_title(name)
    plt.colorbar(im, label="d( log p(y\|x)) / dw")
    plt.tight_layout()
    plt.show()


def plot_heatmap(
    grads_d,
    tokens,
    labels=get_labels("SNLI"),
    names=None,
    center=True,
    **kwargs,
):
    if names is None:
        names = list(grads_d.keys())
    fig, axes = plt.subplots(1, len(names), figsize=(12, 6))
    if len(names) == 1:
        axes = [axes]
    vmin = vmax = None
    if center:
        low = min(np.min(grads_d[name]) for name in names)
        hi = max(np.max(grads_d[name]) for name in names)
        m = max(abs(low), abs(hi))
        vmin, vmax = -m, m
    for name, ax in zip(names, axes):
        grads = grads_d[name]
        im = ax.imshow(grads.T, vmin=vmin, vmax=vmax, **kwargs)
        ax.set_yticks(np.arange(len(tokens)))
        ax.set_yticklabels(tokens)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=90)
        ax.set_title(name)
    plt.colorbar(im, label="d(-log p(y\|x)) / dw")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    args = parse_args()
    run_attribution(args)
