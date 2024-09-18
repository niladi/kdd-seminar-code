from os.path import exists, join
from os import mkdir
import pandas as pd
from matplotlib import pyplot as plt


class Exporter:
    def __init__(self, path: str) -> None:
        self.path = path

    def to_latex(self, df: pd.DataFrame, name: str, label: str) -> None:
        df = df.round(2)
        path = join(self.path, "latex_eval_tables")

        if not exists(path):
            mkdir(path)

        tex = f"""
            \\begin{{table*}}[h!]
	            \\centering
	            \\caption{{All {label} scores}}
	            {df.to_latex()}
	            \\label{{tab:{label}}}
            \\end{{table*}}
        """

        with open(join(path, name), "w") as file:
            file.write(df.to_latex())

        print("LaTeX table has been saved to ", join(path, name))

    def plt_to_png(self, name: str):
        path = join(self.path, "media")
        if not exists(path):
            mkdir(path)
        plt.savefig(join(path, f"{name}.png"), bbox_inches="tight")
