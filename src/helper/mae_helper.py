import argparse
from typing import Dict

import numpy as np

from src.utils.utils import read_json
from joblib import load


class MAEHelper:
    def __init__(self, pred_path: str, gt_path: str, *args, **kwargs):
        self.pred_path = pred_path
        self.gt_path = gt_path

    def compute_mae(self, verbose: bool = False):
        """Compute the Mean Absolute Error between prediction and ground truth.
        It takes into consideration the periodicity of the angles, meaning that
        the MAE is computed as:
            MAE = min(abs(pred - gt), 360 - abs(pred - gt))
        """
        preds, gt = read_json(self.pred_path), read_json(self.gt_path)
        scores_mae = self.compute_mae_from_dict(preds, gt)
        self.print_scores(scores_mae, verbose)
        return scores_mae

    @staticmethod
    def compute_mae_from_dict(
        pred_angles: Dict, gt_angles: Dict, mean: bool = False
    ) -> Dict:
        """
        Compute the MAE between the predicted angles and the ground truth angles.
        :param pred_angles: dictionary with the predicted angles
        :param gt_angles: dictionary with the ground truth angles
        :param mean: whether to return the mean over the angles or not
        :return: a dictionary with the MAE for each angle
        """
        assert len(pred_angles) == len(gt_angles)
        scores_mae = {}
        angles = list(pred_angles[list(pred_angles.keys())[0]]["angles"].keys())
        matrix_preds, matrix_gt = MAEHelper.convert_dict_to_array(
            pred_angles
        ), MAEHelper.convert_dict_to_array(gt_angles)
        scores = MAEHelper.mae(matrix_preds, matrix_gt)
        for index, score in enumerate(scores):
            scores_mae[angles[index]] = score
        if mean:
            scores_mae = {
                key: np.mean(mae_scores) for key, mae_scores in scores_mae.items()
            }
        return scores_mae

    @staticmethod
    def compute_mae_by_rna(pred_angles: Dict, gt_angles: Dict):
        """
        Compute the MAE RNA by RNA
        """
        assert len(pred_angles) == len(gt_angles)
        scores_mae = {}
        angles = list(pred_angles[list(pred_angles.keys())[0]]["angles"].keys())
        matrix_preds, matrix_gt = MAEHelper.convert_dict_to_array(
            pred_angles
        ), MAEHelper.convert_dict_to_array(gt_angles)
        scores = MAEHelper.mae(matrix_preds, matrix_gt, all_mean=False).reshape(
            len(pred_angles), len(angles), -1
        )
        scores = np.nanmean(scores, axis=-1)
        names = list(pred_angles.keys())
        for index, score in enumerate(scores):
            name = names[index]
            scores_mae[name] = score
        return scores_mae

    def print_scores(self, scores_mae: Dict, verbose: bool) -> Dict:
        """
        Print the MAE score for each angle
        :param scores_mae: dictionary with the MAE for each angle
        :return:
        """
        scores = {key: np.mean(mae_scores) for key, mae_scores in scores_mae.items()}
        if not verbose:
            return scores
        to_print = "\t" * 7 + "MAE scores:\n" + "-" * 130 + "\n" + "| "
        for key, score in scores.items():
            to_print += f"{key}: {score:.2f} | "
        to_print += "\n" + "-" * 130 + "\n"
        print(to_print)
        return scores

    @staticmethod
    def convert_dict_to_array(
        preds_dict: Dict, padding_length: int = 512
    ) -> np.ndarray:
        """
        Convert the dictionary of angles to a tensor
        """
        batch_size = len(preds_dict)
        preds = np.stack(
            [
                np.array(
                    MAEHelper.add_padding(preds_dict[k]["angles"][a], padding_length)
                )
                for k in preds_dict
                for a in preds_dict[k]["angles"]
            ]
        )
        seq_len = preds.shape[-1]
        preds = preds.reshape(batch_size, seq_len, -1)
        return preds

    @staticmethod
    def add_padding(preds: np.ndarray, padding_length: int = 200) -> np.ndarray:
        """
        Add padding to the predictions
        """
        preds_padded = [np.nan for _ in range(padding_length)]
        preds_padded[: len(preds)] = preds
        return preds_padded

    @staticmethod
    def mae(
        pred_angles: np.ndarray, gt_angles: np.ndarray, all_mean: bool = True
    ) -> np.ndarray:
        """
        Compute the Mean Absolute Error between two angles, using:
                MAE = min(abs(pred - gt), 360 - abs(pred - gt))
        :param pred_angles: the predicted angles
        :param gt_angles: the true value of the angles
        """
        score_mae = np.stack(
            (abs(pred_angles - gt_angles), (360 - abs(pred_angles - gt_angles)))
        )
        min_mae = score_mae.min(axis=0)
        if not all_mean:
            return min_mae
        score_mae_all_angles = np.nanmean(min_mae, axis=(0, 1))
        return score_mae_all_angles

    @staticmethod
    def custom_mae(pred_angles: np.ndarray, gt_angles: np.ndarray, lr_path: str):
        score_mae = np.stack(
            (abs(pred_angles - gt_angles), (360 - abs(pred_angles - gt_angles)))
        )
        min_mae = score_mae.min(axis=0)
        score_mae_all_angles = np.nanmean(min_mae, axis=(0))
        reg_model = load(lr_path)
        score_mae_all_angles = reg_model.predict(score_mae_all_angles.reshape(1, -1))[0]
        return score_mae_all_angles

    @staticmethod
    def get_arguments():
        parser = argparse.ArgumentParser()
        parser.add_argument("--pred_path", type=str, required=True)
        parser.add_argument("--gt_path", type=str, required=True)
        args = parser.parse_args()
        return args


if __name__ == "__main__":
    args = MAEHelper.get_arguments()
    helper = MAEHelper(**vars(args))
    helper.compute_mae()
