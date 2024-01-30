from typing import List

import pandas as pd

from src.evaluation.score_abstract import ScoreAbstract


class EnrichmentScore(ScoreAbstract):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _compute_score(
        self, metric: str, energy: str, df: pd.DataFrame, name: str
    ) -> float:
        """
        Compute the Enrichment Score for the given metric and energy
        :param metric:
        :param energy:
        :return:
        """
        sorted_metrics = df.sort_values(
            by=metric, ascending=(metric not in self.ascending_metrics)
        ).index.tolist()
        sorted_energy = df.sort_values(
            by=energy, ascending=(energy not in self.ascending_energies)
        ).index.tolist()
        top_n_metrics = self.get_top_n(sorted_metrics, 0.1)
        top_n_energy = self.get_top_n(sorted_energy, 0.1)
        if top_n_energy == top_n_metrics:
            return 10
        intersection = self.get_intersection(top_n_metrics, top_n_energy)
        n_total = len(df) - 1  # remove the native
        es = 100 * intersection / n_total
        return es

    def get_top_n(self, inputs: List, top_n: float) -> List:
        """
        Return the top n% of the list
        :param inputs:
        :param top_n:
        :return:
        """
        return inputs[: int(len(inputs) * top_n)]

    def get_intersection(self, lst1: List, lst2: List) -> int:
        """
        Return the number of intersected elements
        """
        common_lst = list(set(lst1) & set(lst2))
        return len(common_lst)
