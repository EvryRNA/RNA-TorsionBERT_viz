import os

import pandas as pd

from src.enums.enum import (
    LIST_ALL_METRICS,
    LIST_METRICS_WITHOUT_MAE,
    LIST_ENERGIES_WITH_MAE,
)
from src.evaluation.evaluation_helper import EvaluationHelper
from src.helper.decoys import Decoys


class VizDecoysCorrelation(Decoys):
    def __init__(self, pred_out_path: str, pred_out_path_save: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scores = {
            key.replace(".csv", ""): pd.read_csv(
                os.path.join(pred_out_path, key), index_col=0
            )
            for key in os.listdir(pred_out_path)
            if key not in ["3df3A", "3f1hA"]
        }
        self.pred_out_path_save = pred_out_path_save
        os.makedirs(pred_out_path_save, exist_ok=True)

    def run(self):
        df_mae_gold = self.get_mae_custom(self.native_json)
        df_mae_dna = self.get_mae_custom(self.dnabert_json, skip_native=False)
        for rna_name in df_mae_gold["rna_name"].unique():
            c_df_mae_gold = df_mae_gold[df_mae_gold["rna_name"] == rna_name]
            c_df_mae_dna = df_mae_dna[df_mae_dna["rna_name"] == rna_name]
            c_df_mae_gold = self._clean_df(c_df_mae_gold, "mae_gold")
            c_df_mae_dna = self._clean_df(c_df_mae_dna, "mae_dna")
            new_df = pd.concat(
                [self.scores[rna_name], c_df_mae_gold, c_df_mae_dna], axis=1
            )
            new_df.to_csv(os.path.join(self.pred_out_path_save, rna_name + ".csv"))

    def plot_metrics(self, csv_folder: str, save_path: str):
        eval_helper = EvaluationHelper(
            LIST_ALL_METRICS,
            LIST_ALL_METRICS,
            csv_folder,
            save_path=save_path,
            metrics_to_metrics=True,
        )
        all_scores = eval_helper.compute_all_scores(add_mean=False, paper_format=False)
        eval_helper.show_all_scores(all_scores=all_scores, show=False, add_mean=False)

    def plot_energies(self, csv_folder: str, save_path: str):
        eval_helper = EvaluationHelper(
            LIST_METRICS_WITHOUT_MAE,
            LIST_ENERGIES_WITH_MAE,
            csv_folder,
            save_path=save_path,
            metrics_to_metrics=False,
        )
        all_scores = eval_helper.compute_all_scores(add_mean=False, paper_format=False)
        eval_helper.show_all_scores(all_scores=all_scores, show=False, add_mean=False)

    def _clean_df(self, df: pd.DataFrame, new_name: str):
        df["model_name"] = df["model_name"].apply(lambda x: f"normalized_{x}" + ".pdb")
        df.drop(columns=["rna_name"], inplace=True)
        df.set_index("model_name", inplace=True)
        df.rename(columns={"mae": new_name}, inplace=True)
        return df

    @staticmethod
    def compute_correlation_test_set(test_index: int):
        """
        Compute the correlation for the given index of Test Set.
        :param test_index: either 1, 2 or 3. It corresponds to Decoy Test Set I, II or III
        """
        conversion_dict = {1: "I", 2: "II", 3: "III"}
        converted_index = conversion_dict.get(test_index, None)
        if converted_index is None:
            raise ValueError("test_index must be either 1, 2 or 3")
        params = {
            "native_json_path": os.path.join(
                "data", "DECOYS", "angles", "native", f"test_set_{converted_index}.json"
            ),
            "preds_json_path": os.path.join(
                "data", "DECOYS", "pdb", f"TestSet{converted_index}"
            ),
            "pred_out_path": os.path.join(
                "data", "DECOYS", "scores", f"TestSet{converted_index}"
            ),
            "pred_out_path_save": os.path.join(
                "data", "DECOYS", "scores", f"TestSet{converted_index}_MAE"
            ),
            "rna_torsion_pred_path": os.path.join(
                "data",
                "DECOYS",
                "angles",
                "rna_torsionbert",
                f"test_set_{converted_index}.json",
            ),
        }
        viz_decoys = VizDecoysCorrelation(**params)
        viz_decoys.run()
        csv_folder = os.path.join(
            "data", "DECOYS", "scores", f"TestSet{converted_index}_MAE"
        )
        save_path = os.path.join(
            "data", "DECOYS", "results", f"TestSet{converted_index}", "corr_metrics"
        )
        viz_decoys.plot_metrics(csv_folder, save_path)
        save_path = os.path.join(
            "data", "DECOYS", "results", f"TestSet{converted_index}", "corr_energies"
        )
        viz_decoys.plot_energies(csv_folder, save_path)

    @staticmethod
    def plot_pm_decoy_set():
        params = {
            "native_json_path": os.path.join(
                "data", "DECOYS", "angles", "native", "pm_decoys.json"
            ),
            "preds_json_path": os.path.join("data", "DECOYS", "pdb", "PM_DECOY_SET"),
            "pred_out_path": os.path.join("data", "DECOYS", "scores", "PM_DECOY_SET"),
            "pred_out_path_save": os.path.join(
                "data", "DECOYS", "scores", "PM_DECOY_SET_MAE"
            ),
            "rna_torsion_pred_path": os.path.join(
                "data", "DECOYS", "angles", "rna_torsionbert", "pm_decoys.json"
            ),
        }
        viz_decoys = VizDecoysCorrelation(**params)
        viz_decoys.run()
        csv_folder = os.path.join("data", "DECOYS", "scores", "PM_DECOY_SET_MAE")
        save_path = os.path.join(
            "data", "DECOYS", "results", "PM_DECOY_SET", "corr_metrics"
        )
        viz_decoys.plot_metrics(csv_folder, save_path)
        save_path = os.path.join(
            "data", "DECOYS", "results", "PM_DECOY_SET", "corr_energies"
        )
        viz_decoys.plot_energies(csv_folder, save_path)

    @staticmethod
    def plot_rna_puzzles():
        params = {
            "native_json_path": os.path.join(
                "data", "DECOYS", "angles", "native", "rna_puzzles_decoys.json"
            ),
            "preds_json_path": os.path.join("data", "DECOYS", "pdb", "RNA_PUZZLES"),
            "pred_out_path": os.path.join("data", "DECOYS", "scores", "RNA_PUZZLES"),
            "pred_out_path_save": os.path.join(
                "data", "DECOYS", "scores", "RNA_PUZZLES_MAE"
            ),
            "rna_torsion_pred_path": os.path.join(
                "data", "DECOYS", "angles", "rna_torsionbert", "rna_puzzles_decoys.json"
            ),
        }
        viz_decoys = VizDecoysCorrelation(**params)
        viz_decoys.run()
        csv_folder = os.path.join("data", "DECOYS", "scores", "RNA_PUZZLES_MAE")
        save_path = os.path.join(
            "data", "DECOYS", "results", "RNA_PUZZLES", "corr_metrics"
        )
        viz_decoys.plot_metrics(csv_folder, save_path)
        save_path = os.path.join(
            "data", "DECOYS", "results", "RNA_PUZZLES", "corr_energies"
        )
        viz_decoys.plot_energies(csv_folder, save_path)

    @staticmethod
    def plot_near_native():
        params = {
            "native_json_path": os.path.join(
                "data", "DECOYS", "angles", "native", "near_native.json"
            ),
            "preds_json_path": os.path.join("data", "DECOYS", "pdb", "NEAR_NATIVE"),
            "pred_out_path": os.path.join("data", "DECOYS", "scores", "NEAR_NATIVE"),
            "pred_out_path_save": os.path.join(
                "data", "DECOYS", "scores", "NEAR_NATIVE_MAE"
            ),
            "rna_torsion_pred_path": os.path.join(
                "data", "DECOYS", "angles", "rna_torsionbert", "near_native.json"
            ),
        }
        viz_decoys = VizDecoysCorrelation(**params)
        viz_decoys.run()
        csv_folder = os.path.join("data", "DECOYS", "scores", "NEAR_NATIVE_MAE")
        save_path = os.path.join(
            "data", "DECOYS", "results", "NEAR_NATIVE", "corr_metrics"
        )
        viz_decoys.plot_metrics(csv_folder, save_path)
        save_path = os.path.join(
            "data", "DECOYS", "results", "NEAR_NATIVE", "corr_energies"
        )
        viz_decoys.plot_energies(csv_folder, save_path)
