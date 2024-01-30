import os
from typing import Dict, List

import numpy as np
import pandas as pd
from plotly.subplots import make_subplots

import plotly.express as px
import plotly.graph_objects as go

from src.utils.utils import read_json
from src.utils.viz_utils import rename_angles, update_fig_box_plot, clean_fig


class Stats:
    def __init__(self, data_paths: Dict, read_sequence: bool = False):
        self.data_paths = data_paths
        self.data = self.read_data(data_paths)
        if read_sequence:
            self.df_sequences = self.read_sequences(data_paths)

    def read_data(self, data_paths: Dict) -> pd.DataFrame:
        """
        Read the data with angles and return a dataframe
        :param data_paths:
        :return:
        """
        data = {"angles": [], "dataset": [], "angle_names": [], "sequence_len": []}
        for name, path in data_paths.items():
            data_dict = read_json(path)
            for rna_id, rna_data in data_dict.items():
                sequence = len(rna_data["sequence"]) if "sequence" in rna_data else "NA"
                for angle_name, angle_values in rna_data["angles"].items():
                    data["angles"].extend(angle_values)
                    data["dataset"].extend([name] * len(angle_values))
                    data["angle_names"].extend([angle_name] * len(angle_values))
                    data["sequence_len"].extend([sequence] * len(angle_values))
        df = pd.DataFrame(data)
        return df

    def read_data_scatter(self, data_paths: Dict) -> pd.DataFrame:
        data = {"dataset": [], "sequence": []}
        for name, path in data_paths.items():
            data_dict = read_json(path)
            for rna_id, rna_data in data_dict.items():
                sequence = rna_data["sequence"]
                for angle_name, angles in rna_data["angles"].items():
                    index_max = min(len(sequence), len(angles))
                    c_angles, c_sequence = angles[:index_max], sequence[:index_max]
                    c_angles = [
                        angle + 360 if angle < 0 else angle for angle in c_angles
                    ]
                    if angle_name not in data:
                        data[angle_name] = []
                    data[angle_name].extend(c_angles)
                data["dataset"].extend([name] * len(c_sequence))
                data["sequence"].extend(c_sequence)
        df = pd.DataFrame(data)
        return df

    def read_sequences(self, data_paths: Dict) -> pd.DataFrame:
        data = {"dataset": [], "sequence": [], "sequence_len": []}
        for name, path in data_paths.items():
            data_dict = read_json(path)
            for rna_id, rna_data in data_dict.items():
                sequence = rna_data["sequence"] if "sequence" in rna_data else "NA"
                data["dataset"].append(name)
                data["sequence"].append(sequence)
                data["sequence_len"].append(len(sequence))
        df = pd.DataFrame(data)
        return df

    def plot_distribution(self):
        df = self.data.copy()
        df.loc[
            (df["dataset"] == "CASP-RNA") | (df["dataset"] == "RNA-Puzzles"), "dataset"
        ] = "Test"
        df["angles"] = df.apply(
            lambda x: x["angles"] + 360 if x["angles"] < 0 else x["angles"], axis=1
        )
        df = rename_angles(df)
        df = df.rename(columns={"angles": "Angles", "dataset": "Dataset"})
        fig = px.histogram(
            df,
            barmode="overlay",
            x="Angles",
            color="Dataset",
            log_y=False,
            facet_col="angle_names",
            facet_col_wrap=3,
            nbins=150,
            histnorm="percent",
        )
        fig.update_traces(marker_line_width=1, marker_line_color="black")
        fig.update_yaxes(matches=None)
        fig = update_fig_box_plot(fig)
        fig = clean_fig(fig)
        for annotation in fig["layout"]["annotations"]:
            annotation["text"] = annotation["text"].replace("angle_names=", "")
        name_y_axis = "Percent of count"
        fig["layout"]["yaxis4"]["title"] = name_y_axis
        fig["layout"]["xaxis2"]["title"] = "Angle (degree)"
        fig.update_annotations(font_size=40, yshift=5)
        fig.update_layout(margin=dict(b=30, t=30, r=0, l=10))
        fig.write_image(
            "data/output/plots/data_distribution.png", scale=2, width=1200, height=800
        )

    def get_stats(self):
        df_stat = {
            "dataset": [],
            "mean": [],
            "median": [],
            "min": [],
            "max": [],
            "count": [],
        }
        for split in self.df_sequences["dataset"].unique():
            c_df = self.df_sequences[self.df_sequences["dataset"] == split]
            df_stat["dataset"].append(split)
            df_stat["mean"].append(c_df["sequence_len"].mean())
            df_stat["median"].append(c_df["sequence_len"].median())
            df_stat["min"].append(c_df["sequence_len"].min())
            df_stat["max"].append(c_df["sequence_len"].max())
            df_stat["count"].append(len(c_df))
        df_stat = pd.DataFrame(df_stat)
        return df_stat

    def plot_polar_distribution(self):
        path_to_save = os.path.join("img", "polar_distribution.png")
        self.plot_polar(path_to_save, to_show=True)

    def plot_polar(self, name: str, to_show: bool = True):
        df = self.data.copy()
        df.loc[
            (df["dataset"] == "CASP-RNA") | (df["dataset"] == "RNA-Puzzles"), "dataset"
        ] = "Test"
        df["angles"] = df.apply(
            lambda x: x["angles"] + 360 if x["angles"] < 0 else x["angles"], axis=1
        )
        df = rename_angles(df)
        df = df.rename(columns={"angles": "Angles", "dataset": "Dataset"})
        frequencies_dict = self.convert_df_to_frequencies(df)
        angles = df["angle_names"].unique().tolist()
        fig = self._plot_polar(frequencies_dict, angles, colors=None, to_show=to_show)
        fig.write_image(name, scale=2, width=1200, height=800)

    def _plot_polar(
        self, frequencies_dict: dict, angles: list, colors=None, to_show: bool = False
    ):
        fig = self.get_polat_plot(frequencies_dict, angles, colors=colors)
        fig = clean_fig(fig)
        fig.update_layout(margin=dict(b=20, t=50, r=20, l=50))
        new_polars = {
            f"polar{i}": dict(
                radialaxis=dict(
                    type="log",
                    showline=True,
                    showgrid=True,
                    ticks="outside",
                    linewidth=1,
                    linecolor="black",
                    gridcolor="grey",
                    gridwidth=1,
                    dtick=1,
                ),
                angularaxis=dict(
                    linewidth=2, visible=True, linecolor="black", showline=True
                ),
                radialaxis_tickfont_size=12,
                bgcolor="white",
            )
            for i in range(1, 10)
        }
        if to_show:
            fig.update_layout(
                font_size=16,
            )
            fig.update_layout(
                **new_polars,
                showlegend=False,
            )
            fig.show()
        fig.update_layout(
            **new_polars,
            showlegend=False,
        )
        fig.update_layout(paper_bgcolor="white")
        fig.update_layout(
            font_size=16,
        )
        return fig

    def get_polat_plot(self, frequencies_dict: Dict, angles: List, colors=None):
        colors = (
            ["#2C3639", "#3468C0", "#7D0A0A", "#EAD196"] if colors is None else colors
        )
        fig = make_subplots(
            rows=3,
            cols=3,
            subplot_titles=angles,
            specs=[[{"type": "polar"}] * 3] * 3,
        )
        for index, angle in enumerate(angles):
            for split_index, (split, frequencies) in enumerate(
                frequencies_dict.items()
            ):
                fig.add_trace(
                    go.Barpolar(
                        r=frequencies[angle],
                        theta=np.arange(0, 360, 5),
                        marker_color=colors[split_index],
                        marker_line_color="black",
                        marker_line_width=0.5,
                        opacity=0.8,
                        name=split,
                    ),
                    row=(index // 3) + 1,
                    col=(index % 3) + 1,
                )
        return fig

    def convert_df_to_frequencies(self, df: pd.DataFrame):
        output = {}
        for dataset in df["Dataset"].unique():
            c_df = df[df["Dataset"] == dataset]
            output[dataset] = {}
            for angle in df["angle_names"].unique():
                c_angles = c_df[c_df["angle_names"] == angle]
                c_angles = c_angles[c_angles.notnull()]
                angle_values = c_angles["Angles"].values
                angle_values[angle_values < 0] = angle_values[angle_values < 0] + 360
                frequencies = self.get_angles_frequencies(angle_values)
                output[dataset][angle] = frequencies
        return output

    def get_angles_frequencies(self, angles):
        bins = np.arange(0, 360, 5)
        frequencies = np.histogram(angles, bins=bins)[0]
        frequencies = frequencies / np.sum(frequencies)
        return frequencies

    @staticmethod
    def plot_distribution_polar_data(save_path: str):
        prefix = os.path.join("data", "NATIVE", "native")
        data_paths = {
            "Pre-training": os.path.join(prefix, "all_pdb.json"),
            "Training": os.path.join(prefix, "train.json"),
            "Validation": os.path.join(prefix, "valid.json"),
            "RNA-Puzzles": os.path.join(prefix, "rna_puzzles.json"),
            "CASP-RNA": os.path.join(prefix, "casp_rna.json"),
        }
        stats_cli = Stats(data_paths)
        stats_cli.plot_polar(save_path)

    @staticmethod
    def get_data_len():
        prefix = os.path.join("data", "NATIVE", "native")
        data_paths = {
            "All PDB)": os.path.join(prefix, "all_pdb.json"),
            "Training": os.path.join(prefix, "train.json"),
            "Validation": os.path.join(prefix, "valid.json"),
            "RNA-Puzzles": os.path.join(prefix, "rna_puzzles.json"),
            "CASP-RNA": os.path.join(prefix, "casp_rna.json"),
        }
        stats_cli = Stats(data_paths, read_sequence=True)
        df_stats = stats_cli.get_stats()
        print(df_stats)
