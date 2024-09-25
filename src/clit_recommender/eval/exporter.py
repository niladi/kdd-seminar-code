from os.path import exists, join
from os import mkdir
import pandas as pd
from matplotlib import pyplot as plt


class Exporter:
    def __init__(self, path: str) -> None:
        self.path = path

    def to_latex(self, df: pd.DataFrame, name: str) -> None:
        df_round = df.round(2)
        path = join(self.path, "latex_eval_tables")

        if not exists(path):
            mkdir(path)

        with open(join(path, name), "w") as f:
            f.write(
                "\\begin{tabular}{r|" + "".join(["c"] * len(df_round.columns)) + "}\n"
            )
            f.write("\\hline\n")
            f.write(
                " & "
                + " & ".join([f"\\rot{{{x}}}" for x in df_round.columns])
                + " \\\\\n"
            )
            f.write("\\hline\n")
            for i, row in df_round.iterrows():
                s = i if type(i) == str else ""
                f.write(
                    s + " & " + " & ".join([str(x) for x in row.values]) + " \\\\\n"
                )
            f.write("\\hline\n")
            f.write("\\end{tabular}")

        print("LaTeX table has been saved to ", join(path, name))

    def plt_to_png(self, name: str):
        path = join(self.path, "media")
        if not exists(path):
            mkdir(path)
        plt.savefig(join(path, f"{name}.png"), bbox_inches="tight")
