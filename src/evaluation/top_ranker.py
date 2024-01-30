from typing import Dict

import numpy as np
import pandas as pd

from src.evaluation.score_abstract import ScoreAbstract


class TopRanker(ScoreAbstract):
    def __init__(self, top_n: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agg_fn = np.sum
        self.top_n = top_n

    def _compute_score(
        self, metric: str, energy: str, df: pd.DataFrame, name: str
    ) -> float:
        """
        Compute the Enrichment Score for the given metric and energy
        :param metric:
        :param energy:
        :return:
        """
        native_name = f"normalized_{name}.pdb"
        sorted_energy = df.sort_values(
            by=energy, ascending=(energy not in self.ascending_energies)
        )[energy]
        sorted_names = sorted_energy.index.tolist()
        try:
            rank = sorted_names.index(native_name)
        except ValueError:
            rank = sorted_names.index(f"{name}.pdb")
        if rank < self.top_n:
            return 1
        return 0

    def convert_dict_to_df(self, scores: Dict, add_mean: bool = False) -> pd.DataFrame:
        """
        Convert the dictionary of scores to a dataframe by using the mean
        """
        new_scores = {
            energy: [
                self.agg_fn(scores[energy][metric])
                for metric in list(scores[energy].keys())
            ]
            for energy in list(scores.keys())
        }
        new_scores = pd.DataFrame(new_scores).iloc[0, :]
        return new_scores
