# %%
import operator
import os
from typing import List


import torch
from torch import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm, trange
from transformers import get_linear_schedule_with_warmup
from clit_recommender.util import empty_cache

from domain.clit_result import Mention
from process.evaluation import Evaluation
from clit_recommender.config import Config
from data.dataset import ClitRecommenderDataset, DataRow
from process.inference import ClitRecommeder
from domain.metrics import Metrics, MetricsHolder


def train():
    config = Config()
    path = os.path.join(config.cache_dir, "results", config.experiment_name)

    os.makedirs(path)

    with open(os.path.join(path, "config.json"), "w") as f:
        f.write(config.to_json())

    processor = ClitRecommeder(config)
    evaluator = Evaluation(processor)
    model = processor.get_model()

    metrics_holder = MetricsHolder()

    batch: List[DataRow]

    eval = list(ClitRecommenderDataset(config, end=100))
    train = DataLoader(
        ClitRecommenderDataset(config, start=100),
        batch_size=None,
    )

    optimizer = AdamW(model.parameters(), lr=1e-5, eps=1e-8)
    gradient_accumulation_steps: int = 1
    num_warmup_steps: int = 10

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
                list(map(operator.itemgetter(2), output)),
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
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

        metrics = sum(map(evaluator.process_batch, tqdm(eval)), Metrics.zeros())
        metrics_holder.set_metrics_to_last_epoch(metrics)

        print(metrics)


train()

# %%
