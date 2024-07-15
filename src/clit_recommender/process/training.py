# %%
import operator
import os
from typing import List

from numpy import mean
import torch
import random

from torch import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm, trange
from transformers import get_linear_schedule_with_warmup

from clit_recommender.config import Config
from clit_recommender.util import empty_cache
from clit_recommender.data.dataset import ClitRecommenderDataset, DataRow
from clit_recommender.domain.clit_result import Mention
from clit_recommender.domain.metrics import Metrics, MetricsHolder
from clit_recommender.process.evaluation import Evaluation
from clit_recommender.process.inference import ClitRecommeder


def train():
    config = Config(depth=1)
    path = os.path.join(config.cache_dir, "results", config.experiment_name)

    os.makedirs(path)

    with open(os.path.join(path, "config.json"), "w") as f:
        f.write(config.to_json())

    processor = ClitRecommeder(config)
    evaluator = Evaluation(processor)
    model = processor.get_model()

    for param in model.parameters():
        param.requires_grad = True

    metrics_holder = MetricsHolder()

    batch: List[DataRow]
    data_loader = list(DataLoader(ClitRecommenderDataset(config), batch_size=None))
    random.seed(config.seed)
    random.shuffle(data_loader)

    eval = data_loader[:100]
    train = data_loader[100:]
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
                output = processor.process_batch(batch[0])
                loss = output.loss

                if gradient_accumulation_steps >= 1:
                    loss = loss / gradient_accumulation_steps

            loss = loss.mean()
            total_loss += loss.item()
            metrics_holder.add_loss_to_last_epoch(loss.item())
            scaler.scale(loss).backward()

            if step % 500 == 49:
                print()
                print(f"Loss: {total_loss / step}", end="\r")

            if (step + 1) % gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

        metrics_result = Metrics.zeros()
        metrics_prediction = Metrics.zeros()
        for (i,) in tqdm(eval):
            res = evaluator.process_data_row(i)
            metrics_result += res[0]
            metrics_prediction += res[1]

        metrics_holder.set_result_metrics_to_last_epoch(metrics_result)
        metrics_holder.set_prediction_metrics_to_last_epoch(metrics_prediction)

        print("Metrics Result")
        print(metrics_result.get_summary())

        print("Metrics Prediction")
        print(metrics_prediction.get_summary())

        f1 = mean(
            list(
                map(
                    lambda x: x.get_f1(),
                    metrics_holder.get_best_epoch().prediction_metrics.values(),
                )
            )
        )
        if metrics_prediction.get_f1() > f1:
            print("New Model is Best")
            metrics_holder.set_last_epoch_as_best()

        with open(os.path.join(path, "metrics.json"), "w") as f:
            f.write(metrics_holder.to_json())


if __name__ == "__main__":
    train()
