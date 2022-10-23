import argparse
import collections
import os
import json
from pathlib import Path
import random

import numpy as np
import torch

from src.utils import data_utils, logging, metrics, model_utils, tokenizers
from src.grammars import pcfg, spcfg


logger = logging.get_logger(__name__)

TRAINERS = {"pcfg": pcfg.PCFGTrainer(), "spcfg": spcfg.SynchPCFGTrainer()}


def parse_args(use_args=None, namespace=None):
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", type=str, default="output/scratch")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--data_seed", type=int, default=13)

    # Models
    parser.add_argument("--model_type", type=str, choices=["pcfg", "spcfg"])
    parser.add_argument(
        "--load_from",
        type=str,
        default=None,
        help="Path to a .pt file containing model weights",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Whether to save the model parameters",
    )

    # Word tokenizer
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=20000,
        help="Number of words in the vocabulary",
    )
    parser.add_argument(
        "--min_frequency",
        type=int,
        default=3,
        help="Minimum frequency to include a word in the vocabulary",
    )
    parser.add_argument(
        "--load_tokenizer_from",
        type=str,
        default=None,
        help="If non-empty, use a pretrained HuggingFace tokenizer",
    )

    # Data
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--cache_dir", type=str, default="cache")
    parser.add_argument("--overwrite_cache", action="store_true")
    parser.add_argument("--train_on", type=str, default="")
    parser.add_argument("--validate_on", type=str, default="")
    parser.add_argument("--eval_on", type=str, nargs="+", default=[])
    parser.add_argument("--eval_on_train_datasets", nargs="+", default=[])
    parser.add_argument("--max_train_examples", type=int, default=None)
    parser.add_argument("--max_dev_examples", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_eval_length", type=int, default=None)
    parser.add_argument("--min_length", type=int, default=4)
    parser.add_argument(
        "--use_product_length",
        type=int,
        default=None,
        help="For sentence pairs, define length as len(a) x len(b)",
    )

    # Training
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--eval_every", type=int, default=1024)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--criterion", type=str, default="loss")
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--eval_before_training", action="store_true")

    # Learning rate
    parser.add_argument("--optimizer", type=str, default="adamw")
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.1)

    # Grammar
    parser.add_argument("--preterminals", type=int, default=64)
    parser.add_argument("--nonterminals", type=int, default=32)
    parser.add_argument("--state_dim", type=int, default=256)
    parser.add_argument("--use_grad_eval", type=int, default=None)

    # misc
    parser.add_argument("--resume", action="store_true")

    args = parser.parse_args(args=use_args, namespace=namespace)

    if not Path(args.output_dir).exists():
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if not args.validate_on:
        args.validate_on = args.train_on or args.eval_on

    if not args.eval_on:
        args.eval_on = [args.train_on]

    if args.use_grad_eval is None:
        args.use_grad_eval = int(args.model_type in ("spcfg", "pcfg"))

    if args.use_product_length is None:
        args.use_product_length = int(args.model_type == "spcfg")

    if args.load_from and not args.load_tokenizer_from:
        p = Path(args.load_from)
        d = p.parent if p.is_file() else p
        args.load_tokenizer_from = str(d / "tokenizer.json")

    args.remove_special = args.model_type == "pcfg"

    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_train(args, train_dataset, dev_datasets, eval_datasets):
    set_seed(args.seed)

    if args.model_type not in TRAINERS:
        raise NotImplementedError(args.model_type)
    trainer = TRAINERS[args.model_type]
    tokenizer, model = trainer.initialize(args)

    model.to(model_utils.device())
    if torch.distributed.is_initialized():
        model = torch.nn.parallel.DistributedDataParallel(
            model, find_unused_parameters=False
        )
        train_dataset = data_utils.distributed_dataset(train_dataset)
        dev_datasets = {
            k: data_utils.distributed_dataset(v)
            for k, v in dev_datasets.items()
        }
        if eval_datasets:
            eval_datasets = {
                k: data_utils.distributed_dataset(v)
                for k, v in eval_datasets.items()
            }

    if args.load_from:
        trainer.load_from(args, args.load_from, model)

    train_results = trainer.train(
        args, model, tokenizer, train_dataset, dev_datasets
    )
    model_utils.sync_processes()

    eval_results = None
    if eval_datasets:
        if args.save:
            logger.info(f"loading best checkpoint")
            trainer.load_from(args, args.output_dir, model)
        print("CKP " + ("end" if args.train_on else ""))
        eval_results, _ = trainer.evaluate(
            args,
            model,
            tokenizer,
            eval_datasets,
            ckp="end" if args.train_on else "",
        )
        model_utils.sync_processes()

    return train_results, eval_results


def run_eval(args, eval_datasets):
    logger.info(f"run eval")
    if args.model_type not in TRAINERS:
        raise NotImplementedError(args.model_type)
    trainer = TRAINERS[args.model_type]
    tokenizer, model = trainer.initialize(args)
    if args.load_from:
        trainer.load_from(args, args.load_from, model)
    model.to(model_utils.device())
    if torch.distributed.is_initialized():
        model = torch.nn.parallel.DistributedDataParallel(model)
        if eval_datasets:
            for name in eval_datasets:
                eval_datasets[name] = data_utils.distributed_dataset(
                    eval_datasets[name]
                )
    eval_results, _ = trainer.evaluate(
        args, model, tokenizer, eval_datasets, ckp=""
    )
    return eval_results


def get_datasets(args, tokenizer):
    train_dataset, dev_datasets, eval_datasets = None, None, {}
    if args.train_on:
        train_dataset = data_utils.load_dataset(
            args.train_on,
            "train",
            tokenizer,
            max_examples=args.max_train_examples,
            **vars(args),
        )
        dev_datasets = {
            args.validate_on: data_utils.load_dataset(
                args.validate_on,
                "dev",
                tokenizer,
                max_examples=args.max_dev_examples,
                **vars(args),
            )
        }
    for name in args.eval_on:
        eval_datasets[name] = data_utils.load_dataset(
            name,
            "" if (name.endswith(".json") or name.endswith(".tsv")) else "dev",
            tokenizer,
            **vars(args),
        )
    for name in args.eval_on_train_datasets:
        eval_datasets[f"{name}_train"] = data_utils.load_dataset(
            name, "train", tokenizer, **vars(args)
        )

    return train_dataset, dev_datasets, eval_datasets


def run(args):
    tokenizer = tokenizers.get_tokenizer(**vars(args))
    train_dataset, dev_datasets, eval_datasets = get_datasets(args, tokenizer)

    if args.model_type not in TRAINERS:
        raise NotImplementedError(args.model_type)

    if args.train_on:
        run_train(args, train_dataset, dev_datasets, eval_datasets)
    else:
        run_eval(args, eval_datasets)


if __name__ == "__main__":
    args = parse_args()

    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 0))
    if local_world_size > 1:
        rank = int(os.environ.get("RANK", 0))
        torch.cuda.set_device(rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
    else:
        rank = 0

    logging.initialize(args.output_dir, resume=args.resume, rank=rank)
    logger.info(f"logging to {args.output_dir}")
    logger.info(f"args: {vars(args)}")
    if args.train_on and model_utils.is_main_process():
        if args.resume:
            fn = logging.get_resume_name(args.output_dir, s="args.json")
        else:
            fn = Path(args.output_dir) / "args.json"
        with open(fn, "w") as f:
            json.dump(vars(args), f, indent=2)

    run(args)
