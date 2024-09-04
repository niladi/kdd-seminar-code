from dataclasses import field
from os import listdir
from os.path import isdir, join
from typing import Any, Iterable
from ipywidgets.widgets import Layout
from IPython.display import display


from dataclasses_json import config
import torch


def display_with_layout(*widgets_list):
    for widget in widgets_list:
        widget.layout = Layout(width="90%")
    display(*widgets_list)


def enum_list_default(enum_type):  # pylint: disable=invalid-field-call
    return field(
        default_factory=list,
        metadata=config(
            decoder=lambda enum_list: [
                enum_type.__bases__[0].from_dict(entry) for entry in enum_list
            ]
        ),
    )


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


def create_hot_vector(indices, length):
    # Initialize a vector of zeros
    one_hot = ["0"] * length

    # Set the positions in the indices list to 1
    for idx in indices:
        one_hot[idx] = "1"

    # Convert list to a string
    return "".join(one_hot)
