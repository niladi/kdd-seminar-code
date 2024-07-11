# %%
import operator
import os
from typing import List

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
    data_loader = list(
        DataLoader(ClitRecommenderDataset(config, start=100), batch_size=None)
    )
    random.seed(500)
    random.shuffle(data_loader)

    eval = data_loader[:100]
    train = data_loader[100:]
    optimizer = AdamW(model.parameters(), lr=1e-3, eps=1e-8)
    gradient_accumulation_steps: int = 4
    num_warmup_steps: int = 4

    total_steps = len(train) * config.epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_steps / gradient_accumulation_steps,
    )

    scaler = GradScaler()

    for i in trange(config.epochs):
        metrics_holder.add_epoch()
        empty_cache(config.device)
        optimizer.zero_grad()
        model.train()
        total_loss = 0.0

        for step, batch in tqdm(enumerate(train), total=len(train)):

            output = processor.process_batch(batch=batch)
            loss = torch.tensor(
                list(map(lambda x: x.loss, output)),
                dtype=torch.float32,
                requires_grad=True,
            ).to(config.device)

            if gradient_accumulation_steps >= 1:
                loss = loss / gradient_accumulation_steps

            loss = loss.mean()
            total_loss += loss.item()
            metrics_holder.add_loss_to_last_epoch(loss.item())
            scaler.scale(loss).backward()

            if (step + 1) % gradient_accumulation_steps == 0:
                print("Loss", loss.item())
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

        metrics = sum(map(evaluator.process_batch, tqdm(eval)), Metrics.zeros())
        metrics_holder.set_metrics_to_last_epoch(metrics)

        print(metrics.get_summary())


if __name__ == "__main__":
    train()
