from abc import abstractmethod
from typing import Dict, List

import numpy as np
import pandas as pd


class ScoreAbstract:
    """Abstract class for the computation and iteration over the dfs of scores and metrics"""

    def __init__(self, ascending_metrics: List[str], ascending_energies: List[str]):
        self.ascending_metrics = ascending_metrics
        self.ascending_energies = ascending_energies
        self.agg_fn = np.mean

    def compute_score(
        self,
        energy_list: List,
        metrics_list: List,
        dfs: Dict[str, pd.DataFrame],
        add_mean: bool = False,
    ) -> pd.DataFrame:
        """
        Loop to compute the score
        :return:
        """
        scores: Dict = {
            energy: {metric: [] for metric in metrics_list} for energy in energy_list
        }
        for energy in energy_list:
            for metric in metrics_list:
                for name, df in dfs.items():
                    try:
                        current_score = (self._compute_score(metric, energy, df, name),)
                    except KeyError:
                        current_score = np.nan  # type: ignore
                    if not np.isnan(current_score):
                        scores[energy][metric].append(current_score)
        new_scores = self.convert_dict_to_df(scores, add_mean=add_mean)
        return new_scores

    @abstractmethod
    def _compute_score(
        self, metric: str, energy: str, df: pd.DataFrame, name: str
    ) -> float:
        raise NotImplementedError

    def convert_dict_to_df(self, scores: Dict, add_mean: bool = False) -> pd.DataFrame:
        """
        Convert the dictionary of scores to a dataframe by using the mean
        """
        new_scores = {
            energy: [
                round(self.agg_fn(scores[energy][metric]), 2)
                for metric in list(scores[energy].keys())
            ]
            for energy in list(scores.keys())
        }
        metrics_list = list(scores[list(scores.keys())[0]].keys())
        new_scores = pd.DataFrame(new_scores, index=metrics_list)
        if add_mean:
            new_scores.loc["Mean"] = round(new_scores.mean(axis=0), 2)  # type: ignore
        return new_scores
