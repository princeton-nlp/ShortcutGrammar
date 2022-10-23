import collections
import contextlib
from dataclasses import asdict
import json
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, BatchSampler, RandomSampler
from tqdm import tqdm

from src.utils import data_utils, logging, metrics, model_utils
from src.utils.model_utils import to_device

logger = logging.get_logger(__name__)


def save(args, model, ckp):
    name = "model.pt"
    fn = Path(args.output_dir) / name
    params = dict(list(model.named_parameters()))
    state_dict = collections.OrderedDict()
    for k, v in model.state_dict().items():
        if k in params and params[k].requires_grad:
            state_dict[k] = v
    logger.info(
        f"saving {'model'} to {fn} "
        f"({len(state_dict)}/{len(params)} parameters)"
    )
    torch.save(state_dict, str(fn))


def asdict_(d):
    if type(d) == dict:
        return d
    return asdict(d)


class Trainer:
    def add_arguments(self, parser):
        return

    def initialize(self, args):
        raise NotImplementedError

    def get_predictions(self, args, model, batch, outputs, tokenizer, **kwargs):
        raise NotImplementedError

    def score_predictions(self, args, predictions, **kwargs):
        raise NotImplementedError

    def get_inputs(self, args, batch, **kwargs):
        raise NotImplementedError

    def get_optimizer(self, args, model, **kwargs):
        params = [p for p in model.parameters() if p.requires_grad]
        if args.optimizer == "adamw":
            return AdamW(
                params,
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
            )
        raise NotImplementedError(args.optimizer)

    def checkpoint(
        self,
        args,
        model,
        tokenizer,
        eval_datasets,
        ckp,
        best=None,
        save_predictions=False,
        base_report=None,
    ):
        report, predictions = self.evaluate(
            args,
            model,
            tokenizer,
            eval_datasets,
            ckp=ckp,
            base_report=base_report,
        )

        if save_predictions and model_utils.is_main_process():
            name = ckp.split(".")[-1]
            logger.info(f"saving {name} predictions")
            with open(Path(args.output_dir) / f"best.{name}.json", "w") as f:
                json.dump(best, f, indent=2)
            for dataset, preds in predictions.items():
                dataset_name = str(Path(dataset).with_suffix("")).replace(
                    "/", "_"
                )
                with open(
                    Path(args.output_dir)
                    / f"best_predictions.{name}.{dataset_name}.json",
                    "w",
                ) as f:
                    json.dump(predictions, f, indent=2)

        if best is None:
            return best, False, report

        if args.criterion in ("loss", "x_nll"):
            compare = lambda a, b: a < b
        else:
            compare = lambda a, b: a > b

        if args.criterion not in best or compare(
            report[args.criterion], best[args.criterion]
        ):
            best["ckp"] = ckp
            best["patience"] = 0
            best.update(report)
            logger.info(f"new best: {best}")
            if model_utils.is_main_process():
                with open(Path(args.output_dir) / f"best.json", "w") as f:
                    json.dump(best, f, indent=2)
                for dataset, preds in predictions.items():
                    with open(
                        Path(args.output_dir)
                        / f"best_predictions.{dataset}.json",
                        "w",
                    ) as f:
                        json.dump(predictions, f, indent=2)
                if args.save:
                    save(args, model, ckp)
        else:
            best["patience"] += 1

        report["patience"] = best["patience"]

        stop_early = False
        if (
            args.patience not in (None, -1)
            and best["patience"] >= args.patience
        ):
            stop_early = True
            logger.info(
                f"{args.patience} checkpoints with no improvement, stopping"
            )

        model_utils.sync_processes()
        return best, stop_early, report

    def evaluate_one(
        self,
        args,
        model,
        tokenizer,
        dataset,
        eval_dataloader,
        ckp="",
        base_report=None,
        device=None,
    ):
        model.eval()
        logger.info(f"evaluating on {dataset}")
        try:
            epoch = int(ckp.split(".")[0])
        except:
            epoch = -1

        # Torch-Struct uses backpropagation for some operations so don't
        # use `torch.no_grad`
        if args.use_grad_eval:
            requires_grad = set()
            for n, p in model.named_parameters():
                if p.requires_grad:
                    requires_grad.add(n)
                p.requires_grad = False

        ctx = contextlib.nullcontext if args.use_grad_eval else torch.no_grad
        with ctx():
            t = tqdm(eval_dataloader, desc=f"eval [{ckp}]")
            predictions = []
            eval_loss = 0
            for step, batch in enumerate(t):
                inputs = to_device(self.get_inputs(args, batch))
                inputs["is_prediction"] = True
                inputs["epoch"] = epoch
                outputs = model(**inputs)
                predictions += self.get_predictions(
                    args, model, batch, outputs, tokenizer
                )
                loss = outputs["loss"].mean()
                # This just clears up some memory--no parameter updates.
                if args.use_grad_eval:
                    loss.backward()
                    model.zero_grad()
                eval_loss += loss.item()
                t.set_postfix({"loss": loss.item()})
            eval_loss = eval_loss / len(eval_dataloader)
            logger.info(f"avg eval loss: {eval_loss}")

        if args.use_grad_eval:
            for n, p in model.named_parameters():
                if n in requires_grad:
                    p.requires_grad = True

        if torch.distributed.is_initialized():
            logger.info(f"gathering predictions")
            predictions = model_utils.all_gather_list(predictions)

        report, predictions = self.score_predictions(args, predictions)
        if base_report:
            report.update(base_report)
        logger.info(f"{dataset} results at {ckp}: {report}")

        if model_utils.is_main_process():
            logger.info(f"writing {dataset} results to {args.output_dir}")
            name = str(Path(dataset).with_suffix("")).replace("/", "_")
            pckp = f"{ckp}." if (ckp and not ckp[0].isdigit()) else ""
            with open(
                Path(args.output_dir) / f"metrics.{name}.{pckp}json", "w"
            ) as f:
                json.dump(report, f, indent=2)
            with open(
                Path(args.output_dir) / f"predictions.{name}.{pckp}json", "w"
            ) as f:
                json.dump(predictions, f, indent=2)
        model_utils.sync_processes()
        return report, predictions

    def evaluate(
        self, args, model, tokenizer, eval_datasets, ckp="", base_report=None
    ):
        logger.info("evaluating")
        eval_dataloaders = {}
        for dataset in eval_datasets:
            eval_dataloader = DataLoader(
                eval_datasets[dataset],
                batch_size=args.eval_batch_size,
                collate_fn=data_utils.collate,
            )
            eval_dataloaders[dataset] = eval_dataloader
        reports = {}
        predictions = {}
        for dataset, dataloader in eval_dataloaders.items():
            reports[dataset], predictions[dataset] = self.evaluate_one(
                args, model, tokenizer, dataset, dataloader, ckp, base_report
            )
        report = metrics.average_dicts(list(reports.values()), short=True)
        if len(reports) > 1:
            report.update(reports)
        if model_utils.is_main_process():
            pckp = f"{ckp}." if (ckp and not ckp[0].isdigit()) else ""
            logger.info(f"writing average results to {args.output_dir}")
            with open(Path(args.output_dir) / f"metrics.{pckp}json", "w") as f:
                json.dump(report, f, indent=2)
        model_utils.sync_processes()
        return report, predictions

    def load_from(self, args, path, model, device=None, **kwargs):
        if str(path).endswith(".pt"):
            fn = path
        else:
            fns = sorted(
                Path(path).glob("model*"),
                key=lambda p: p.lstat().st_ctime,
            )
            if len(fns) == 0:
                raise ValueError(f"no model.pt in {path}")
            fn = fns[-1]
        if device is None:
            device = model_utils.device()
        logger.info(f"loading checkpoint from {fn}")
        state_dict = torch.load(fn, map_location=device)
        state_dict = {
            k.replace("module.", ""): v for k, v in state_dict.items()
        }
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        logger.info(f"{len(missing)} missing, {len(unexpected)} unexpected")
        logger.info(f"Missing: {missing}")
        logger.info(f"Unexpected: {unexpected}")
        return model

    def train(self, args, model, tokenizer, train_dataset, dev_datasets):
        if model_utils.is_main_process():
            logger.info(f"writing args and tokenizer to {args.output_dir}")
            with open(Path(args.output_dir) / "args.json", "w") as f:
                json.dump(vars(args), f, indent=2)
            if hasattr(tokenizer, "save"):
                tokenizer.save(Path(args.output_dir) / "tokenizer.json")

        sampler = BatchSampler(
            RandomSampler(train_dataset), args.train_batch_size, drop_last=False
        )

        train_dataloader = DataLoader(
            train_dataset,
            batch_sampler=sampler,
            collate_fn=data_utils.collate,
        )

        model.to(model_utils.device())

        optimizer = self.get_optimizer(args, model)
        scheduler = None
        steps_per_epoch = (
            len(train_dataloader) / args.gradient_accumulation_steps
        )
        num_steps = int(max(args.steps, args.epochs * steps_per_epoch))
        if args.steps > (args.epochs * steps_per_epoch):
            args.epochs = int(args.steps // steps_per_epoch) + 1
        logger.info(f"training for {num_steps} steps / {args.epochs} epochs")

        best = {"ckp": "", "patience": 0}
        stop_early = False
        global_step = 0

        if args.eval_before_training:
            logger.info(f"evaluating before training")
            best, _, _ = self.checkpoint(
                args,
                model,
                tokenizer,
                dev_datasets,
                ckp="0.0",
                best=best,
                base_report={"ckp": "0.0"},
            )

        epoch_losses = []
        did_opt_step_eval = False
        for epoch in range(args.epochs):
            epoch_loss = 0
            opt_step = 0
            checkpoint_loss = 0
            logger.info(f"epoch: {epoch}")
            t = tqdm(train_dataloader, desc=f"train [{epoch}]")
            for step, batch in enumerate(t):
                model.train()
                inputs = to_device(self.get_inputs(args, batch))
                inputs["epoch"] = epoch
                inputs["step"] = step
                outputs = model(**inputs)
                loss = outputs["loss"].mean() / args.gradient_accumulation_steps
                epoch_loss += loss.item()
                checkpoint_loss += loss.item()
                t.set_postfix(
                    {"loss": loss.item() * args.gradient_accumulation_steps}
                )

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    loss.backward()
                    global_step += 1
                    opt_step += 1
                    if args.max_grad_norm:
                        nn.utils.clip_grad_norm_(
                            model.parameters(),
                            args.max_grad_norm,
                            error_if_nonfinite=True,
                        )
                    optimizer.step()
                    model.zero_grad()
                    optimizer.zero_grad()
                else:
                    if torch.distributed.is_initialized():
                        with model.no_sync():
                            loss.backward()
                    else:
                        loss.backward()

                if (
                    args.eval_every > 0
                    and opt_step > 0
                    and opt_step % args.eval_every == 0
                    and not did_opt_step_eval
                ):
                    did_opt_step_eval = True
                    ckp = f"{epoch}.{int((epoch * steps_per_epoch) + step)}"
                    checkpoint_loss = checkpoint_loss / args.eval_every
                    logger.info(f"training loss (ckp {ckp}): {checkpoint_loss}")
                    base_report = {"training_loss": checkpoint_loss, "ckp": ckp}
                    best, stop_early, report = self.checkpoint(
                        args,
                        model,
                        tokenizer,
                        dev_datasets,
                        ckp,
                        best,
                        base_report=base_report,
                    )

                    new_best = best["ckp"] == ckp
                    checkpoint_loss = 0
                    if stop_early:
                        break

                if global_step > num_steps:
                    logger.info(
                        f"global_step {global_step} > num_steps {num_steps}, "
                        "stopping"
                    )
                    stop_early = True
                    break

            if stop_early:
                logger.info(f"stop early set to true somewhere, stopping")
                break

            logger.info(f"end of epoch {epoch}")
            epoch_loss = epoch_loss / opt_step
            epoch_losses.append(epoch_loss)
            logger.info(f"average training loss: {epoch_loss}")
            ckp = f"{epoch}.{(epoch + 1) * len(train_dataloader)}"
            base_report = {"training_loss": epoch_loss, "ckp": ckp}
            best, stop_early, report = self.checkpoint(
                args,
                model,
                tokenizer,
                dev_datasets,
                ckp,
                best,
                base_report=base_report,
            )
            new_best = best["ckp"] == ckp
            if stop_early:
                break
        return best
