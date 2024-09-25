from collections import defaultdict
from multiprocessing import freeze_support
import operator
import os
from typing import List, Tuple, Type


from numpy import mean
import torch
import random

from torch import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm, trange
from transformers import get_linear_schedule_with_warmup

from clit_recommender.data.best_graphs.factory import BestGraphFactory
from clit_recommender.data.dataset.clit_recommender_data_set import (
    ClitRecommenderDataSet,
    ClitRecommenderDynamicBatchDataSet,
)
from clit_recommender.data.embeddings_precompute import EmbeddingsPrecompute
from clit_recommender.domain.clit_mock.combined_node import (
    CombinedNode,
    IntersectionNode,
    MajorityVoting,
    UnionNode,
)
from clit_recommender.domain.datasets import Dataset, DatasetSplitType
from clit_recommender.domain.systems import System
from clit_recommender.config import Config
from clit_recommender.eval.exporter import Exporter
from clit_recommender.eval.heatmap import create_systems_x2_used
from clit_recommender.eval.plot import create_amount_of_systems_plot
from clit_recommender.util import empty_cache
from clit_recommender.domain.data_row import DataRow
from clit_recommender.domain.metrics import Metrics, MetricsHolder
from clit_recommender.eval.evaluation import Evaluation
from clit_recommender.process.inference import ClitRecommeder


def _offline_data(config: Config):
    print("Checking Offline Data.")

    offline_data = EmbeddingsPrecompute(config)
    if not offline_data.exists():
        print("Precomputed Data Missing. Creating ... ")
        offline_data.generate_uri_to_idx()
        offline_data.generate_text_embeddings()

    best_graph_factory = BestGraphFactory(config)

    if not best_graph_factory.exists():
        print("Best Graphs not exists. Creating ...")
        best_graph_factory.create()
    print("Offline Data. Done.")


def cross_train(
    config: Config,
    training_sets: List[Dataset],
    eval_sets: List[Dataset],
    save: bool = True,
):
    config.datasets = training_sets + eval_sets

    _offline_data(config)

    path = os.path.join(
        config.results_dir, "experiments_cross_domain", config.experiment_name
    )
    if save:
        os.makedirs(path)
    config.datasets = training_sets
    if save:
        with open(os.path.join(path, "config_train.json"), "w") as f:
            f.write(config.to_json())

    train = list(
        DataLoader(ClitRecommenderDynamicBatchDataSet(config), batch_size=None)
    )
    random.shuffle(train)
    config.datasets = eval_sets
    if save:
        with open(os.path.join(path, "config_eval.json"), "w") as f:
            f.write(config.to_json())
    eval = list(DataLoader(ClitRecommenderDynamicBatchDataSet(config), batch_size=None))
    random.shuffle(eval)

    _train(config, train, eval, path, save)


def train_full(config: Config, save: bool = True, plots: bool = False):
    _offline_data(config)

    path = os.path.join(config.results_dir, "experiments", config.experiment_name)

    if save:
        os.makedirs(path)

        with open(os.path.join(path, "config.json"), "w") as f:
            f.write(config.to_json())

    train = list(
        DataLoader(
            ClitRecommenderDynamicBatchDataSet(config, DatasetSplitType.TRAIN),
            batch_size=None,
        )
    )
    random.shuffle(train)

    eval = ClitRecommenderDynamicBatchDataSet(config, DatasetSplitType.EVAL)

    _train(config, train, eval, path, save, plots)


def _train(
    config: Config,
    train: List[List[DataRow]],
    eval: List[List[DataRow]],
    path: str,
    save: bool = True,
    plots: bool = False,
):

    processor = ClitRecommeder(config)
    evaluator = Evaluation(processor)
    model = processor.get_model()

    for param in model.parameters():
        param.requires_grad = True

    metrics_holder = MetricsHolder()

    batch: List[DataRow]
    optimizer = AdamW(model.parameters(), lr=1e-5, eps=1e-8)
    gradient_accumulation_steps: int = 2
    num_warmup_steps: int = 4

    total_steps = len(train) * config.epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_steps / gradient_accumulation_steps,
    )

    scaler = GradScaler(device=config.device)

    for i in trange(config.epochs):
        metrics_holder.add_epoch()
        empty_cache(config.device)
        optimizer.zero_grad()
        model.train()
        total_loss = 0.0

        for step, batch in tqdm(enumerate(train), total=len(train)):

            with autocast(device_type=config.device):
                for idx, row in enumerate(batch):
                    output = processor.process_batch(row)
                    if idx == 0:
                        loss = output.loss
                    else:
                        loss += output.loss

                loss = loss / len(batch)
            if (
                gradient_accumulation_steps >= 1
            ):  # https://huggingface.co/docs/accelerate/usage_guides/gradient_accumulation
                loss = loss / gradient_accumulation_steps

            loss = loss.mean()
            total_loss += loss.item()
            scaler.scale(loss).backward()

            if step % 500 == 49:
                print()
                print(f"Loss: {total_loss / step}", end="\r")
                metrics_holder.add_loss_to_last_epoch(total_loss / step)

            if (step + 1) % gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

        metrics_result = Metrics.zeros()
        metrics_prediction = Metrics.zeros()

        graphs = []

        for batch in tqdm(eval):
            res = evaluator.process_dynamic_batch(batch)
            metrics_result += res[0]
            metrics_prediction += res[1]
            graphs.append(res[2])

        metrics_holder.set_result_metrics_to_last_epoch(metrics_result)
        metrics_holder.set_prediction_metrics_to_last_epoch(metrics_prediction)

        print("Metrics Result (MD Task)")
        print(metrics_result.get_summary())

        print("Metrics Prediction (Graph prediction Task)")
        print(metrics_prediction.get_summary())

        metric = mean(
            list(
                map(
                    lambda x: x.get_metric(config.metric_type),
                    metrics_holder.get_best_epoch().prediction_metrics.values(),
                )
            )
        )

        if metrics_prediction.get_metric(config.metric_type) > metric:
            print("New Model is Best")
            metrics_holder.set_last_epoch_as_best()

            if plots:
                last_roundet: Tuple[Tuple[float, ...], Type[CombinedNode]] = list(
                    filter(
                        lambda s: s is not None and len(s) > 0,
                        map(lambda x: x.get_last_level_tuple_roundet(), graphs),
                    )
                )
                systems = []
                amount = defaultdict(lambda: defaultdict(int))
                exporter = Exporter(path)
                for sys, combination_type in last_roundet:
                    systems.append(sys)
                    key = {
                        UnionNode: "Union",
                        IntersectionNode: "Intersection",
                        MajorityVoting: "Majority Voting",
                    }.get(combination_type)
                    amount[int(sum(sys))][key] += 1

                create_systems_x2_used(systems, config.metric_type, exporter, True)

                create_amount_of_systems_plot(
                    amount, config.metric_type, exporter, True
                )

        if save:
            with open(os.path.join(path, "metrics.json"), "w") as f:
                f.write(metrics_holder.to_json())

            with open(os.path.join(path, "summary.txt"), "w") as f:
                f.write("Result Metrics\n")
                f.write(
                    next(
                        iter(metrics_holder.get_best_epoch().result_metrics.values())
                    ).get_summary()
                )
                f.write("\nPrediction Metrics\n")
                f.write(
                    next(
                        iter(
                            metrics_holder.get_best_epoch().prediction_metrics.values()
                        )
                    ).get_summary()
                )
        if metrics_holder.best_epoch_index - i > 2:
            print()
            print("No Improvement in 2 epochs. Stopping training.")
            print("Best Epoch: ", metrics_holder.best_epoch_index)
            break
    return next(iter(metrics_holder.get_best_epoch().result_metrics.values()))


if __name__ == "__main__":
    freeze_support()
    train_full(
        Config(
            depth=1,
            datasets=list(Dataset),
            systems=[
                System.BABEFLY,
                System.DBPEDIA_SPOTLIGHT,
                System.OPEN_TAPIOCA,
                System.REFINED_MD_PROPERTIES,
                System.REL_MD_PROPERTIES,
                System.SPACY_MD_PROPERTIES,
                System.TAGME,
                System.TEXT_RAZOR,
            ],
            epochs=20,
        ),
        True,
        True,
    )
