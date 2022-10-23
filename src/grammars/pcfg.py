# PCFG
import random

from nltk.tree import Tree
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torch_struct

from src.utils import data_utils, logging, metrics, model_utils, tokenizers
from src.models.base_trainer import Trainer

logger = logging.get_logger(__name__)


class Res(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.u1 = nn.Linear(H, H)
        self.u2 = nn.Linear(H, H)

        self.v1 = nn.Linear(H, H)
        self.v2 = nn.Linear(H, H)
        self.w = nn.Linear(H, H)

    def forward(self, x):
        x = self.w(x)
        x = x + torch.relu(self.v1(torch.relu(self.u1(x))))
        return x + torch.relu(self.v2(torch.relu(self.u2(x))))


class PCFG(nn.Module):
    # Adapted from https://github.com/harvardnlp/compound-pcfg
    def __init__(
        self, vocab_size=100, preterminals=10, nonterminals=10, state_dim=512
    ):
        super(PCFG, self).__init__()
        self.nonterminals = self.NT = nonterminals
        self.preterminals = self.T = preterminals
        self.all_states = self.S = nonterminals + preterminals
        self.dim = state_dim
        self.state_dim = state_dim

        self.rule_mlp = nn.Linear(state_dim, self.all_states ** 2)
        self.root_mlp = nn.Sequential(
            nn.Linear(state_dim, state_dim),
            Res(state_dim),
            Res(state_dim),
            nn.Linear(state_dim, self.nonterminals),
        )
        self.vocab_mlp = nn.Sequential(
            nn.Linear(state_dim, state_dim),
            Res(state_dim),
            Res(state_dim),
            nn.Linear(state_dim, vocab_size),
        )
        self.V = vocab_size

        self.rule_types = np.arange(self.S).astype(str)

        self.t_emb = nn.Parameter(torch.randn(preterminals, state_dim))
        self.nt_emb = nn.Parameter(torch.randn(nonterminals, state_dim))
        self.root_emb = nn.Parameter(torch.randn(1, state_dim))

    def normalize(self):
        T, NT, S, D = self.T, self.NT, self.S, self.state_dim
        root_log_probs = self.root_mlp(self.root_emb).log_softmax(-1).squeeze(0)
        emission_log_probs = self.vocab_mlp(self.t_emb).log_softmax(-1)
        rule_log_probs = (
            self.rule_mlp(self.nt_emb).log_softmax(-1).view(NT, S, S)
        )
        self.root_log_probs, self.emission_log_probs, self.rule_log_probs = (
            root_log_probs,
            emission_log_probs,
            rule_log_probs,
        )

    def forward(self, **inputs):
        self.normalize()
        x = inputs["input_ids"]
        B, N = x.shape
        T, NT, S, V = self.T, self.NT, self.S, self.V
        idxs = x.unsqueeze(1).expand(B, T, N)
        emissions = self.emission_log_probs.unsqueeze(0).expand(B, T, V)
        terms = torch.gather(emissions, -1, idxs).permute(0, 2, 1).contiguous()
        potentials = (
            terms,
            self.rule_log_probs.unsqueeze(0).expand(B, NT, S, S),
            self.root_log_probs.unsqueeze(0).expand(B, NT),
        )
        dist = torch_struct.SentCFG(potentials, lengths=inputs["length"])
        log_probs = dist.partition
        return {"log_probs": log_probs, "loss": -log_probs, "dist": dist}

    @property
    def device(self):
        return self.t_emb.device


def sample_tree(dist):
    return dist._struct(torch_struct.SampledSemiring).marginals(
        dist.log_potentials, lengths=dist.lengths
    )


def to_tree(tokens, terms, spans, nonterminals, rule_types=None):
    lst = []
    for w, k in zip(tokens, terms.nonzero()[:, 1]):
        s = k.item()
        label = str(s + nonterminals)
        if rule_types is not None:
            label = rule_types[s + nonterminals]
        lst.append(Tree(label, [w]))
    for w, i, nt in spans.nonzero():
        s = nt.item()
        label = str(s)
        if rule_types is not None:
            label = rule_types[s]
        cur = lst[i] = lst[i + w + 1] = Tree(label, [lst[i], lst[i + w + 1]])
    return cur


def add_arguments(parser):
    pass


class PCFGTrainer(Trainer):
    def initialize(self, args):
        tokenizer = tokenizers.get_tokenizer(**vars(args))
        model = PCFG(
            vocab_size=len(tokenizer),
            preterminals=args.preterminals,
            nonterminals=args.nonterminals,
            state_dim=args.state_dim,
        )
        return tokenizer, model

    def get_predictions(self, args, model, batch, outputs, tokenizer):
        if torch.distributed.is_initialized():
            model = model.module
        predictions = []
        log_probs = outputs["log_probs"].cpu().tolist()
        loss = outputs["loss"].cpu().tolist()
        terms, rules, init, spans = outputs["dist"].argmax
        for i, e in enumerate(batch["examples"]):
            p = {"e": e, "log_probs": log_probs[i], "loss": loss[i]}
            try:
                tree = to_tree(
                    data_utils.tokens(batch["input_ids"][i], tokenizer),
                    terms[i],
                    spans[i],
                    model.NT,
                    rule_types=model.rule_types,
                )
                p["tree"] = str(tree.pformat(margin=1000))
            except Exception as e:
                logger.warning(str(e))
                p["tree"] = ""
                raise e
            predictions.append(p)
        return predictions

    def score_predictions(self, args, predictions):
        trees = "\n".join(
            [
                p["tree"]
                for p in random.sample(predictions, k=min(4, len(predictions)))
            ]
        )
        logger.info(f"trees:\n{trees}")
        return metrics.score_predictions(predictions)

    def get_inputs(self, args, batch, **kwargs):
        if "a" in batch:
            return model_utils.concatenate_pair(batch)
        inputs = {
            k: batch[k] for k in ("input_ids", "attention_mask", "length")
        }
        return inputs
