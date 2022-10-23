# Transformer classifier
import argparse
from collections import Counter
from dataclasses import asdict, dataclass, field
import json
import math
from typing import Any
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer
from tqdm import tqdm

from src.utils import data_utils, logging, metrics, model_utils, tokenizers
from src.models.base_trainer import Trainer

logger = logging.get_logger(__name__)


class LSTMClassifier(nn.Module):
    def __init__(self, args, tokenizer):
        super().__init__()
        config = AutoConfig.from_pretrained(args.model_name_or_path)
        encoder = AutoModel.from_pretrained(
            args.model_name_or_path, config=config
        )
        encoder.resize_token_embeddings(len(tokenizer))
        idxs = torch.arange(len(tokenizer))
        with torch.no_grad():
            emb = encoder.embeddings.word_embeddings(idxs)
        self.dim = emb.shape[-1]
        self.lstm = nn.LSTM(
            input_size=self.dim,
            hidden_size=self.dim,
            num_layer=2,
            batch_first=2,
            dropout=0.5,
            bidirectional=True,
        )
        self.head = nn.Linear(self.dim, args.num_labels)
        self.args = args

    def forward(self, **inputs):
        labels = inputs.pop("label") if "label" in inputs else None
        model_inputs = {k: inputs[k] for k in ("input_ids", "attention_mask")}
        if "position_ids" in inputs:
            model_inputs["position_ids"] = inputs["position_ids"]
        model_outputs = self.model(**model_inputs, return_dict=True)
        logits = self.head(model_outputs.last_hidden_state[:, 0])
        if "biased_log_probs" in inputs and not inputs.get("is_prediction"):
            logits = inputs["biased_log_probs"] + logits.log_softmax(-1)
        log_probs = F.log_softmax(logits, dim=-1)

        outputs = {"log_probs": log_probs}
        if labels is not None:
            outputs["loss"] = -torch.gather(
                log_probs, 1, labels.unsqueeze(-1)
            ).squeeze(-1)
        if "weight" in inputs:
            loss = outputs["loss"] * inputs["weight"]
            # outputs["uw_loss"] = outputs["loss"]
            outputs["loss"] = loss
        return outputs


class Classifier(nn.Module):
    def __init__(self, args, tokenizer):
        super().__init__()
        config = AutoConfig.from_pretrained(args.model_name_or_path)
        self.model = AutoModel.from_pretrained(
            args.model_name_or_path, config=config
        )
        self.model.resize_token_embeddings(len(tokenizer))
        if args.whitespace_tokenizer == "reinit":
            logger.info(f"reinitializing token embeddings")
            self.model._init_weights(self.model.embeddings.word_embeddings)
            # for module in self.model.embeddings.modules():
            #     self.model._init_weights(module)
        self.head = nn.Linear(self.model.config.hidden_size, args.num_labels)
        self.args = args

    @property
    def device(self):
        return self.head.weight.device

    def forward(self, **inputs):
        labels = inputs.pop("label") if "label" in inputs else None
        model_inputs = {k: inputs[k] for k in ("input_ids", "attention_mask")}
        if "position_ids" in inputs:
            model_inputs["position_ids"] = inputs["position_ids"]
        model_outputs = self.model(**model_inputs, return_dict=True)
        logits = self.head(model_outputs.last_hidden_state[:, 0])
        if "biased_log_probs" in inputs and not inputs.get("is_prediction"):
            logits = inputs["biased_log_probs"] + logits.log_softmax(-1)
        log_probs = F.log_softmax(logits, dim=-1)

        outputs = {"log_probs": log_probs}
        if labels is not None:
            outputs["loss"] = -torch.gather(
                log_probs, 1, labels.unsqueeze(-1)
            ).squeeze(-1)
        if "weight" in inputs:
            loss = outputs["loss"] * inputs["weight"]
            # outputs["uw_loss"] = outputs["loss"]
            outputs["loss"] = loss
        return outputs


class ClassifierTrainer(Trainer):
    def initialize(self, args):
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, use_fast=True
        )
        if (
            "tacred" in args.train_on
            or "tacred" in args.eval_on
            and not args.word_tokenizer
        ):
            special = data_utils.tacred_special_tokens()
            tokenizer.add_special_tokens({"additional_special_tokens": special})
        if args.model_type == "classifier":
            model = Classifier(args, tokenizer)
        else:
            raise NotImplementedError(args.model_type)
        model.model.resize_token_embeddings(len(tokenizer))
        return tokenizer, model

    def get_predictions(
        self, args, model, batch, outputs, tokenizer, collapse_labels=False
    ):
        predictions = []
        log_probs = model_utils.to_list(outputs["log_probs"])
        loss = model_utils.to_list(outputs["loss"])
        predicted = model_utils.to_list(
            torch.argmax(outputs["log_probs"], dim=-1)
        )
        for i, e in enumerate(batch["examples"]):
            p = {
                "e": e,
                "predicted": predicted[i],
                "log_probs": log_probs[i],
                "loss": loss[i],
                "score": -loss[i],
            }
            if collapse_labels:
                p["full_log_probs"] = log_probs[i]
                p["log_probs"] = [
                    log_probs[i][0],
                    np.log(np.sum(np.exp(log_probs[i][1:]))).item(),
                ]
                p["predicted"] = np.argmax(p["log_probs"]).item()
                p["loss"] = -p["log_probs"][e["label"]]
            predictions.append(p)
        return predictions

    def score_predictions(self, args, predictions):
        if "tacred" in args.train_on or "tacred" in args.eval_on:
            return metrics.score_tacred_predictions(predictions)
        return metrics.score_classifier_predictions(predictions)

    def get_inputs(self, args, batch, **kwargs):
        if "a" in batch:
            inputs = model_utils.concatenate_pair(batch)
        else:
            inputs = {
                k: batch[k] for k in ("input_ids", "attention_mask", "label")
            }
        if "biased_log_probs" in batch:
            inputs["biased_log_probs"] = batch["biased_log_probs"]
        if args.weighted_loss and "weight" in batch:
            inputs["weight"] = batch["weight"]
        return inputs
