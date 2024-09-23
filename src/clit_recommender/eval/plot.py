# Create a list of labels for the x-axis
from numbers import Number
from typing import Dict, List
from matplotlib import pyplot as plt
from clit_recommender.domain.metrics import MetricType
from clit_recommender.eval.exporter import Exporter


def get_labels(amount: Dict[Number, Dict[str, Number]]) -> List[str]:
    labels = set()
    for _a in amount.values():
        for key in _a.keys():
            labels.add(key)
    return list(labels)


# Create a list of colors for the stacked bars
def get_colors():
    return [
        "#00876C",
        "#4664aa",
        "#fce500",
        "#a3107c",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]


def create_amount_of_systems_plot(
    amount: Dict[Number, Dict[str, Number]],
    metric_type: MetricType,
    exporter: Exporter = None,
    close: bool = False,
):

    # Extract the keys and values from the 'amount' dictionary
    keys = list(amount.keys())
    values = list(amount.values())
    labels = get_labels(amount)
    colors = get_colors()

    # Create a list of bottom values for the stacked bars
    bottom = [0] * len(keys)

    # Plot the stacked bars
    for i in range(len(labels)):
        plt.bar(
            keys,
            [v.get(labels[i], 0) for v in values],
            bottom=bottom,
            label=labels[i],
            color=colors[i],
        )
        bottom = [
            bottom[j] + [v.get(labels[i], 0) for v in values][j]
            for j in range(len(keys))
        ]

    # Add labels and title to the plot
    plt.xlabel("Amount of Systems")
    plt.ylabel("Frequency")
    plt.title("Amount of Systems in Best Pipeline")

    # Add legend to the plot
    plt.legend()

    if exporter:
        exporter.plt_to_png(
            f"amount_of_systems_in_best_pipeline_{metric_type.name.lower()}"
        )

    if close:
        plt.close()
    else:
        plt.show()
