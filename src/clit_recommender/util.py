from os import listdir
from os.path import isdir, join
from typing import Any, Iterable

import json

import torch


def iterate_dirs(path, dir_only=True):
    return filter(
        lambda x: (not dir_only) or isdir(x),
        map(lambda s: join(path, s), listdir(path)),
    )


def flat_map(f, xs):
    return (y for ys in xs for y in f(ys))


def batch_items(iterable: Iterable[Any], n: int = 1):
    """
    Batches an iterables by yielding lists of length n. Final batch length may be less than n.
    :param iterable: any iterable
    :param n: batch size (final batch may be truncated)
    """
    current_batch = []
    for item in iterable:
        current_batch.append(item)
        if len(current_batch) == n:
            yield current_batch
            current_batch = []
    if current_batch:
        yield current_batch


def empty_cache(device):

    if "cuda" in device:
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
        torch.cuda.empty_cache()
    elif "mps" in device:
        # MPS doesn't have an explicit cache emptying method.
        # If needed, manage memory more explicitly here.
        pass
    else:
        # For CPU and other devices, there's no cache to empty.
        pass
