import os
from typing import Dict, List, Optional

import pandas as pd
import plotly.express as px

from src.enums.enum import DICT_TO_CHANGE, ASCENDING_METRICS, ASCENDING_ENERGIES
from src.evaluation.enrichment_score import EnrichmentScore
from src.evaluation.pcc import PCC
from src.evaluation.top_ranker import TopRanker


class EvaluationHelper:
    def __init__(
        self,
        metrics_list: List,
        energy_list: List,
        csv_folder: str,
        save_path: Optional[str] = None,
        metrics_to_metrics: bool = False,
    ):
        self.metrics_list = metrics_list
        self.energy_list = energy_list
        self.dfs = {
            name.replace(".csv", ""): pd.read_csv(
                os.path.join(csv_folder, name), index_col=[0]
            ).rename(columns=DICT_TO_CHANGE)
            for name in os.listdir(csv_folder)
        }
        self.save_path = save_path
        if not os.path.exists(self.save_path):  # type: ignore
            os.makedirs(self.save_path, exist_ok=True)  # type: ignore
        self.ascending_energies = (
            ASCENDING_METRICS if metrics_to_metrics else ASCENDING_ENERGIES
        )
        self.evaluation_scores = {
            "PCC": PCC(ASCENDING_METRICS, self.ascending_energies),
            "ES": EnrichmentScore(ASCENDING_METRICS, self.ascending_energies),
        }
        if not metrics_to_metrics:
            self.evaluation_scores = {
                **self.evaluation_scores,
                **{
                    "TOP-1": TopRanker(1, ASCENDING_METRICS, self.ascending_energies),
                },
            }

    def compute_all_scores(
        self, add_mean: bool = False, paper_format: bool = True
    ) -> Dict:
        all_scores = {}
        for score_name, score_helper in self.evaluation_scores.items():
            all_scores[score_name] = score_helper.compute_score(
                self.energy_list, self.metrics_list, self.dfs, add_mean
            )
            if self.save_path is not None:
                line_term = "\\ \n" if paper_format else "\n"  # type: ignore
                params = (
                    {"sep": "&", "lineterminator": line_term}
                    if paper_format
                    else {"sep": ",", "lineterminator": line_term}
                )
                all_scores[score_name].T.to_csv(
                    os.path.join(self.save_path, f"{score_name}.csv"),
                    **params,
                )
        return all_scores

    def show_all_scores(
        self,
        all_scores: Optional[Dict] = None,
        show: bool = False,
        add_mean: bool = False,
    ):
        """
        Show a heatmap of the different scores
        :return:
        """
        all_scores = (
            self.compute_all_scores(add_mean=add_mean)
            if all_scores is None
            else all_scores
        )
        for score_name, score_df in all_scores.items():
            try:
                score_df = score_df.abs().rename(
                    columns={"BARNABA-eRMSD": "eRMSD"}, index={"BARNABA-eRMSD": "eRMSD"}
                )
            except TypeError:
                score_df = score_df.abs()
            if isinstance(score_df, pd.Series):
                continue
            fig = px.imshow(
                score_df.T,
                text_auto=True,
                aspect="auto",
                zmin=0,
                zmax=1 if score_name == "PCC" else 10,
            )
            if show:
                fig.show()
            if self.save_path:
                fig.write_image(
                    os.path.join(self.save_path, f"{score_name}.png"), scale=2
                )
