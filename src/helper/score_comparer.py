import os
from typing import Dict

import numpy as np
import pandas as pd

from src.helper.mae_helper import MAEHelper
from src.utils.utils import read_json
import plotly.express as px

import plotly.graph_objs as go
from src.utils.viz_utils import clean_fig, update_fig_box_plot, rename_angles

ALL_ANGLES = [
    "alpha",
    "beta",
    "gamma",
    "delta",
    "epsilon",
    "zeta",
    "chi",
    "eta",
    "theta",
]


class ScoreComparer:
    def __init__(self, gt_path: str):
        self.gt_path = gt_path

    def read_data(self, model_paths: Dict) -> pd.DataFrame:
        all_data = {"model": [], "angle": [], "MAE": []}
        for name, path in model_paths.items():
            try:
                data = read_json(path)
                all_data["angle"].extend(list(data.keys()))
                all_data["MAE"].extend(list(data.values()))
                all_data["model"].extend([name] * len(data))
            except NotADirectoryError:
                continue
        df = pd.DataFrame(all_data)
        return df

    def read_json_padding(self, in_path: str, seq_len: int) -> Dict:
        """
        Read the json file and pad the angles to the specified length.
        """
        output = read_json(in_path)
        new_output = {}
        for key, value in output.items():
            new_output[key] = (
                {"sequence": value["sequence"], "angles": {}}
                if "sequence" in value
                else {"angles": {}}
            )
            for angle, angle_values in value["angles"].items():
                pad_values = [np.nan for _ in range(seq_len)]
                index = min(len(angle_values), seq_len)
                pad_values[:index] = angle_values[:index]
                if angle not in new_output[key]["angles"]:
                    new_output[key]["angles"][angle] = pad_values
        return new_output

    def read_data_all(self, model_paths: Dict, add_native=True) -> pd.DataFrame:
        all_data = {"model": [], "angle_name": [], "angle": []}
        if add_native:
            model_paths["native"] = self.gt_path
        for name, path in model_paths.items():
            data = read_json(path)
            angles = self.get_angles_values_from_data(data)
            for angle, angle_values in angles.items():
                all_data["angle"].extend(angle_values)
                all_data["angle_name"].extend([angle] * len(angle_values))
                all_data["model"].extend([name] * len(angle_values))
        df = pd.DataFrame(all_data)
        return df

    def read_data_all_examples(self, model_paths: Dict) -> pd.DataFrame:
        """Read all the data and keep the index of the angles."""
        all_data = {
            "model": [],
            "angle_name": [],
            "index": [],
            "angle": [],
            "native": [],
        }
        native = read_json(self.gt_path)
        for name, path in model_paths.items():
            data = read_json(path)
            for rna_name, all_angles in data.items():
                for angle, angle_values in all_angles["angles"].items():
                    native_angles = native[rna_name]["angles"][angle]
                    if len(native_angles) < len(angle_values):
                        angle_values = angle_values[: len(native_angles)]
                    all_data["native"].extend(native_angles[: len(angle_values)])
                    all_data["model"].extend(len(angle_values) * [name])
                    all_data["angle_name"].extend(len(angle_values) * [angle])
                    all_data["index"].extend(list(range(len(angle_values))))
                    all_data["angle"].extend(angle_values)
        df = pd.DataFrame(all_data)
        return df

    def read_data_mae(self, model_paths: Dict) -> pd.DataFrame:
        """
        Read the data and compute MAE for each angle.
        :param model_paths:
        :return:
        """
        all_data = {
            "model": [],
            "angle": [],
            "MAE": [],
            "index": [],
            "name": [],
            "sequence_len": [],
        }
        native = self.read_json_padding(self.gt_path, 512)
        for name, path in model_paths.items():
            data = self.read_json_padding(path, 512)
            try:
                scores = MAEHelper.compute_mae_by_rna(data, native)
            except AssertionError:
                continue
            for c_name, values in data.items():
                if c_name == "R1138":
                    continue
                len_rna = len(native[c_name]["sequence"])
                for index, (angle, angle_values) in enumerate(values["angles"].items()):
                    mae = scores[c_name][index]
                    all_data["MAE"].append(mae)
                    all_data["model"].append(name)
                    all_data["angle"].append(angle)
                    all_data["index"].append(index)
                    all_data["name"].append(c_name)
                    all_data["sequence_len"].append(len_rna)
        return pd.DataFrame(all_data)

    def get_angles_values_from_data(self, data: Dict) -> Dict:
        angles = {}
        for value in data.values():
            for angle, angle_value in value["angles"].items():
                if angle in angles:
                    angles[angle].extend(angle_value)
                else:
                    angles[angle] = angle_value
        return angles

    def plot_mae_distribution(self, model_paths: Dict):
        df = self.read_data_mae(model_paths)
        fig = px.box(
            df,
            x="model",
            y="MAE",
            color="model",
            facet_col="angle",
            title=f"MAE for RNAPuzzles",
            facet_col_wrap=3,
        )
        fig.update_traces(marker_line_width=1, marker_line_color="black")
        fig = clean_fig(fig)
        fig.show()

    def return_table(self, model_paths: Dict):
        df = self.read_data_mae(model_paths)
        new_df = df.groupby(["model", "angle"]).agg({"MAE": "mean"}).reset_index()
        new_df = new_df.pivot(index="model", columns="angle", values="MAE")
        return new_df

    @staticmethod
    def get_df_mae_all_test():
        prefix = os.path.join("data", "NATIVE", "native")
        paths = {
            "rna_puzzles": os.path.join(prefix, "rna_puzzles.json"),
            "casp_rna": os.path.join(prefix, "casp_rna.json"),
        }
        all_df = []
        for split, gt_path in paths.items():
            model_paths = ScoreComparer.get_model_paths(split)
            score_comparer = ScoreComparer(gt_path)
            df = score_comparer.read_data_mae(model_paths)
            all_df.append(df)
        df = pd.concat(all_df)
        df = df.rename(columns={"model": "Model"})
        df = rename_angles(df)
        fig = px.box(
            df,
            x="angle",
            y="MAE",
            color="Model",
            notched=False,
        )
        fig.update_traces(marker_line_width=1, marker_line_color="black")
        fig = clean_fig(fig)
        fig = update_fig_box_plot(fig)
        for data in fig.data:
            data["marker"] = dict(color="#000000", opacity=1, size=8)
        COLORS = {
            "SPOT-RNA-1D": "#BF3131",
            "RNABERT": "#F6B17A",
            "RNA-TorsionBERT": "#3468C0",
        }
        for cat, color in COLORS.items():
            fig.update_traces(fillcolor=color, selector=dict(name=cat))
        fig.update_layout(
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=0.8,
                font=dict(size=14, color="black"),
            )
        )
        fig.show()
        fig.write_image(
            os.path.join("img", "mae_distribution_all_test.png"),
            scale=2,
            width=800,
            height=300,
        )

    @staticmethod
    def get_table_results():
        prefix = os.path.join("data", "NATIVE", "native")
        paths = {
            "rna_puzzles": os.path.join(prefix, "rna_puzzles.json"),
            "casp_rna": os.path.join(prefix, "casp_rna.json"),
        }
        all_df = []
        for split, gt_path in paths.items():
            model_paths = ScoreComparer.get_model_paths(split, is_rna=False)
            score_comparer = ScoreComparer(gt_path)
            df = score_comparer.read_data_mae(model_paths)
            all_df.append(df)
        df = pd.concat(all_df)
        df = df.rename(columns={"model": "Model"})
        df = rename_angles(df)
        df = (
            df[["Model", "angle", "MAE"]]
            .groupby(["Model", "angle"])
            .mean()
            .reset_index()
        )
        df = df.pivot(index="angle", columns="Model", values="MAE")
        # Add a row with mean values
        df.loc["Mean"] = df.mean()
        print(df)
        # df.to_latex(os.path.join("data", "output", "tables", "mae_table.tex"), escape=False, float_format="%.1f")

    @staticmethod
    def get_model_paths(split: str, is_rna: bool = True):
        prefix = os.path.join("data", "NATIVE")
        model_paths = {
            "RNA-TorsionBERT": os.path.join(prefix, "rna_torsionbert", f"{split}.json"),
            "SPOT-RNA-1D": os.path.join(prefix, "spot_rna_1d", f"{split}.json"),
        }
        return model_paths

    @staticmethod
    def get_len_plot():
        prefix = os.path.join("data", "NATIVE", "native")
        paths = {
            "rna_puzzles": os.path.join(prefix, "rna_puzzles.json"),
            "casp_rna": os.path.join(prefix, "casp_rna.json"),
        }
        all_df = []
        for split, gt_path in paths.items():
            model_paths = ScoreComparer.get_model_paths(split)
            score_comparer = ScoreComparer(gt_path)
            df = score_comparer.read_data_mae(model_paths)
            all_df.append(df)
        df = pd.concat(all_df)
        df = df.rename(columns={"model": "Model"})
        df = rename_angles(df)
        new_df = df.groupby(["Model", "sequence_len"], as_index=False)["MAE"].mean()
        new_df["std"] = df.groupby(["Model", "sequence_len"], as_index=False)[
            "MAE"
        ].std()["MAE"]
        new_df = new_df[new_df["sequence_len"] < 512]
        df_dna = new_df[new_df["Model"] == "RNA-TorsionBERT"]
        df_spot = new_df[new_df["Model"] == "SPOT-RNA-1D"]
        COLORS = {
            "SPOT-RNA-1D": "#BF3131",
            "RNABERT": "#F6B17A",
            "DNABERT-3": "#3468C0",
        }
        score_dnabert = ScoreComparer.scatter_position(
            df_dna, color=COLORS["DNABERT-3"], name="RNA-TorsionBERT"
        )
        score_spot = ScoreComparer.scatter_position(
            df_spot, color=COLORS["SPOT-RNA-1D"], name="SPOT-RNA-1D"
        )
        fig = go.Figure(score_dnabert + score_spot)
        fig.update_traces(marker_line_width=1, marker_line_color="black")
        fig = clean_fig(fig)
        fig.update_layout(
            xaxis_title="Sequence length (nt)",
            yaxis_title="MAE (degrees)",
            font=dict(
                family="Computer Modern",
                size=18,
            ),
        )
        fig.update_layout(
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=0.93,
                xanchor="right",
                x=0.99,
                font=dict(size=14, color="black"),
            )
        )
        fig.show()
        fig.write_image(
            os.path.join("img", "len_plot.png"), width=800, height=600, scale=2
        )

    @staticmethod
    def scatter_position(df: pd.DataFrame, color, name):
        return [
            go.Scatter(
                name=name,
                x=df["sequence_len"],
                y=df["MAE"],
                mode="lines+markers",
                line=dict(color=color),
            ),
            go.Scatter(
                name=f"Upper Bound {name}",
                x=df["sequence_len"],
                y=df["MAE"] + df["std"],
                mode="lines",
                marker=dict(color=color, opacity=0.5),
                line=dict(width=1),
                showlegend=False,
            ),
            go.Scatter(
                name=f"Lower Bound {name}",
                x=df["sequence_len"],
                y=df["MAE"] - df["std"],
                marker=dict(color=color, opacity=0.5),
                line=dict(width=1),
                mode="lines",
                fill="tonexty",
                showlegend=False,
            ),
        ]
