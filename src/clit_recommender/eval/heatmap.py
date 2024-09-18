from typing import List, Tuple

import numpy as np


import matplotlib.pyplot as plt
import seaborn as sns


from matplotlib.colors import LinearSegmentedColormap

import pandas as pd

from clit_recommender.domain.metrics import MetricType
from clit_recommender.domain.systems import System
from clit_recommender.eval.exporter import Exporter


custom_cmap = LinearSegmentedColormap.from_list(
    "custom_cmap", ["#FFFFFF", "#00876C"]
)  # Create custom colormap


def create_systems_x2_used(
    systems: List[Tuple[float, ...]],
    metric_type: MetricType,
    exporter: Exporter = None,
    close: bool = False,
):
    systems_matrix = sum([np.outer(np.array(s), np.array(s)) for s in systems])

    df = pd.DataFrame(systems_matrix)
    labels = [system.label for system in list(System)]
    df.index = labels
    df.columns = labels
    df.index.name = "Systems"

    df = df.loc[:, (df != 0).any(axis=0)]
    df = df.loc[(df != 0).any(axis=1), :]

    # Rescale the numbers by dividing by 1000
    df = df / 1000

    ax = sns.heatmap(
        df, linewidth=0.5, cmap=custom_cmap, annot=True, fmt=".2f", cbar=True
    )

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    # Add labels and title to the plot
    plt.xlabel("System")
    plt.ylabel("System")
    plt.title("Heatmap of System to System " + metric_type.value.lower() + " (in 1000)")

    if exporter:
        exporter.plt_to_png(f"heatmap_of_sys_sys_{metric_type.value.lower()}")

    if close:
        plt.close()
    else:
        plt.show()
