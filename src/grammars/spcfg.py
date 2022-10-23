# SPCFG
import json
import random

from nltk.tree import Tree
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import torch_struct

from src.grammars import scky
from src.utils import data_utils, logging, metrics, tokenizers
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


class SynchPCFG(nn.Module):
    def __init__(
        self,
        vocab_size=100,
        preterminals=10,
        nonterminals=10,
        state_dim=256,
        src_key="a",
        tgt_key="b",
    ):
        super(SynchPCFG, self).__init__()
        self.state_dim = state_dim

        self.t_emb = nn.Parameter(torch.randn(preterminals, state_dim))
        self.t_type_emb = nn.Parameter(torch.randn(preterminals, state_dim))
        self.nt_emb = nn.Parameter(torch.randn(nonterminals, state_dim))
        self.root_emb = nn.Parameter(torch.randn(1, state_dim))
        self.nonterminals = self.NT = NT = nonterminals
        self.preterminals = self.T = T = preterminals
        self.all_states = self.S = S = preterminals + nonterminals
        self.dim = state_dim

        self.rule_mlp = nn.Linear(state_dim, self.all_states ** 2)
        self.root_mlp = nn.Sequential(
            nn.Linear(state_dim, state_dim),
            Res(state_dim),
            Res(state_dim),
            nn.Linear(state_dim, self.nonterminals),
        )
        self.vocab_size = vocab_size

        self.emission_types = ["copy", "w/w", "w/e", "e/w"]
        self.emission_type_mlp = nn.Sequential(
            nn.Linear(state_dim, len(self.emission_types))
        )
        # ww
        self.src_vocab_mlp = nn.Sequential(
            nn.Linear(state_dim, state_dim),
            Res(state_dim),
            Res(state_dim),
            nn.Linear(state_dim, vocab_size),
        )
        self.src_enc_emb = nn.Embedding(vocab_size, state_dim)
        self.tgt_vocab_mlp = nn.Sequential(
            nn.Linear(state_dim, state_dim),
            Res(state_dim),
            Res(state_dim),
            nn.Linear(state_dim, vocab_size),
        )

        self.we_vocab_mlp = nn.Sequential(
            nn.Linear(state_dim, state_dim),
            Res(state_dim),
            Res(state_dim),
            nn.Linear(state_dim, vocab_size),
        )
        self.ew_vocab_mlp = nn.Sequential(
            nn.Linear(state_dim, state_dim),
            Res(state_dim),
            Res(state_dim),
            nn.Linear(state_dim, vocab_size),
        )

        self.src_key = src_key
        self.tgt_key = tgt_key

    def normalize(self):
        self.root_log_probs = F.log_softmax(
            self.root_mlp(self.root_emb), -1
        ).squeeze(0)
        self.src_emission_log_probs = F.log_softmax(
            self.src_vocab_mlp(self.t_emb), -1
        )
        self.we_emission_log_probs = F.log_softmax(
            self.we_vocab_mlp(self.t_emb), -1
        )
        self.ew_emission_log_probs = F.log_softmax(
            self.ew_vocab_mlp(self.t_emb), -1
        )
        NT, S = self.NT, self.S
        self.rule_log_probs = F.log_softmax(
            self.rule_mlp(self.nt_emb), -1
        ).view(NT, S, S)
        self.emission_type_log_probs = F.log_softmax(
            self.emission_type_mlp(self.t_type_emb), -1
        )

    def forward(self, **inputs):
        self.normalize()
        src = inputs[self.src_key]["input_ids"]
        tgt = inputs[self.tgt_key]["input_ids"]
        N = src.size(1)
        M = tgt.size(1)
        B = src.size(0)
        T, NT, S = self.T, self.NT, self.S

        # emission log probabilities
        # (B, N, T)
        src_terms = F.embedding(src, self.src_emission_log_probs.T)
        # (B, N, H)
        src_vocab_emb = self.src_enc_emb(src)
        # (B, N, H, T) -> (B, N, T, H)
        src_emb = (src_vocab_emb.unsqueeze(-1) + self.t_emb.T).transpose(-2, -1)
        # (B, N, T, V)
        tgt_emission_log_probs = self.tgt_vocab_mlp(src_emb).log_softmax(-1)
        # (B, N, T, M)
        tgt_terms = torch.gather(
            tgt_emission_log_probs, -1, tgt.view(B, 1, 1, M).expand(B, N, T, M)
        )

        # rule types
        wcopy, ww, we, ew = self.emission_type_log_probs.T

        # (B, N, T, M) -> (B, N, M, T)
        ww_log_probs = (
            src_terms.unsqueeze(-1) + tgt_terms + ww.view(1, 1, T, 1)
        ).transpose(-2, -1)

        # copy. (B, N, M)
        copy_mask = (src.unsqueeze(-1) != tgt.unsqueeze(-2)) * -1e5
        # (B, N, T, M) -> (B, N, M, T)
        copy_log_probs = (
            copy_mask.unsqueeze(-2) + wcopy.view(1, 1, T, 1)
        ).transpose(-2, -1) + src_terms.unsqueeze(-2)

        # terms[b, i, j, 1, 1, t]: log p(t -> w_i/w_j)
        # terms[b, i, j, 0, 1, t]: log p(t -> empty/w_j)
        # terms[b, i, j, 1, 0, t]: log p(t -> w_i/empty)
        # terms[b, i, j, 0, 0, t]: not used
        terms = torch.zeros(B, N, M, 2, 2, T, device=src.device)
        terms[:, :, :, 1, 1] += torch.logsumexp(
            torch.stack([copy_log_probs, ww_log_probs], -1), -1
        )

        # (B, N, T, V) -> (B, N, T) -> (B, N, 1, T)
        we_log_probs = F.embedding(src, self.we_emission_log_probs.T).unsqueeze(
            -2
        )
        # use the eos token as the null embedding.
        eos_idx = inputs[self.src_key]["length"] - 1
        # (B, N, T, M) -> (B, T, M) -> (B, 1, T, M) -> (B, 1, M, T)
        ew_log_probs = F.embedding(tgt, self.ew_emission_log_probs.T).unsqueeze(
            -3
        )
        terms[:, :, :, 0, 1] += ew_log_probs + ew.view(1, 1, 1, T)
        terms[:, :, :, 1, 0] += we_log_probs + we.view(1, 1, 1, T)
        terms[:, :, :, 0, 0] = -1e5

        potentials = (
            terms,
            self.rule_log_probs.unsqueeze(0).expand(B, NT, S, S),
            self.root_log_probs.unsqueeze(0).expand(B, NT),
        )
        # subtract the bos and eos tokens
        lengths = torch.stack(
            [
                inputs[self.src_key]["length"] - 2,
                inputs[self.tgt_key]["length"] - 2,
            ],
            -1,
        )
        dist = scky.SynchCFG(potentials, lengths=lengths)
        log_probs = dist.partition
        return {"log_probs": log_probs, "loss": -log_probs, "dist": dist}

    @property
    def device(self):
        return self.t_emb.device


def to_tree(a_tokens, b_tokens, terms, spans, nonterminals):
    nodes = {}
    leaves = []
    a_len = 0
    b_len = 0
    for ind, (i, j, wn, wm, s) in enumerate(terms.nonzero().tolist()):
        a = "".join(a_tokens[i : i + wn])
        b = "".join(b_tokens[j : j + wm])
        label = str(s + nonterminals)
        node = Tree(label, [f"{a}/{b}"])
        nodes[(i - 1, j - 1)] = node
        leaves.append(node)
        a_len = max(a_len, i)
        b_len = max(b_len, j)
    lst = []
    for wn, wm, i, j, s in spans.nonzero().tolist():
        l = (i, j)
        r = (i + wn - 1, j + wm - 1)
        label = str(s)
        node = Tree(label, [nodes[l], nodes[r]])
        nodes[l] = nodes[r] = node
        lst.append(node)
    return lst[-1]


class SynchPCFGTrainer(Trainer):
    def initialize(self, args):
        tokenizer = tokenizers.get_tokenizer(**vars(args))
        model = SynchPCFG(
            vocab_size=len(tokenizer),
            preterminals=args.preterminals,
            nonterminals=args.nonterminals,
            state_dim=args.state_dim,
        )
        return tokenizer, model

    def get_predictions(self, args, model, batch, outputs, tokenizer):
        predictions = []
        log_probs = outputs["log_probs"].cpu().tolist()
        loss = outputs["loss"].cpu().tolist()
        try:
            terms, rules, init, spans = [
                v.detach() for v in outputs["dist"].argmax
            ]
        except Exception as e:
            print(batch["examples"])
            raise e
        if torch.distributed.is_initialized():
            model = model.module
        a, b = model.src_key, model.tgt_key
        for i, e in enumerate(batch["examples"]):
            p = {"e": e, "log_probs": log_probs[i], "loss": loss[i]}
            try:
                tree = to_tree(
                    data_utils.tokens(batch[a]["input_ids"][i], tokenizer),
                    data_utils.tokens(batch[b]["input_ids"][i], tokenizer),
                    terms[i],
                    spans[i],
                    model.NT,
                )
                p["tree"] = str(tree.pformat(margin=1000))
            except Exception as e:
                logger.warning(str(e))
                p["tree"] = ""
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
        ks = ("input_ids", "length", "uid")
        inputs = {a: {k: batch[a][k] for k in ks} for a in ("a", "b")}
        return inputs
