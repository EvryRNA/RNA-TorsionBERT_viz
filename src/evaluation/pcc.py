from typing import Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from src.evaluation.score_abstract import ScoreAbstract


class PCC(ScoreAbstract):
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
        sorted_metrics, sorted_energy = df[metric].values, df[energy].values
        sorted_metrics, sorted_energy = self.filter_inf(sorted_metrics, sorted_energy)
        try:
            pearson_corr, _ = pearsonr(sorted_metrics, sorted_energy)
        except ValueError:
            return np.nan
        return abs(pearson_corr)

    def filter_inf(
        self, matrix1: np.ndarray, matrix2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Remove the INF values from the matrix
        :param matrix1:
        :param matrix2:
        :return:
        """
        matrix1 = np.nan_to_num(matrix1, nan=1e301)
        matrix2 = np.nan_to_num(matrix2, nan=1e301)
        indices = np.where((matrix1 < 1e300) & (matrix2 < 1e300))
        new_matrix1 = matrix1[indices]
        new_matrix2 = matrix2[indices]
        return new_matrix1, new_matrix2
