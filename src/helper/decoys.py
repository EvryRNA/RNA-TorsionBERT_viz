import os
from typing import Dict

import pandas as pd

from src.helper.mae_helper import MAEHelper
from src.utils.utils import read_json_padding


class Decoys:
    def __init__(
        self,
        native_json_path: str,
        preds_json_path: str,
        rna_torsion_pred_path: str,
        seq_len: int = 512,
    ):
        self.native_json = read_json_padding(native_json_path, seq_len)
        self.dnabert_json = read_json_padding(rna_torsion_pred_path, seq_len)
        self.preds_json_path = {
            key.replace(".json", ""): read_json_padding(
                os.path.join(preds_json_path, key), seq_len
            )
            for key in os.listdir(preds_json_path)
        }

    def get_mae_custom(self, native_json: Dict, skip_native: bool = True):
        output = {"rna_name": [], "model_name": [], "mae": []}
        native_array = MAEHelper.convert_dict_to_array(native_json)
        for index, rna_name in enumerate(native_json):
            current_preds = self.preds_json_path[rna_name]
            pred_array = MAEHelper.convert_dict_to_array(current_preds)
            for index_method, method in enumerate(current_preds):
                if method == rna_name and skip_native:
                    continue
                maes = MAEHelper.mae(pred_array[index_method], native_array[index])
                output["rna_name"].append(rna_name)
                output["model_name"].append(method)
                output["mae"].append(maes)
        output = pd.DataFrame(output)
        return output

    def get_mae(self):
        output = {"rna_name": [], "model_name": [], "mae": []}
        native_array = MAEHelper.convert_dict_to_array(self.native_json)
        dnabert_array = MAEHelper.convert_dict_to_array(self.dnabert_json)
        for index, rna_name in enumerate(self.native_json):
            current_preds = self.preds_json_path[rna_name]
            pred_array = MAEHelper.convert_dict_to_array(current_preds)
            output["rna_name"].append(rna_name)
            output["model_name"].append("dnabert")
            output["mae"].append(
                MAEHelper.mae(native_array[index], dnabert_array[index])
            )
            for index_method, method in enumerate(current_preds):
                if method == rna_name:
                    continue
                maes = MAEHelper.mae(pred_array[index_method], native_array[index])
                output["rna_name"].append(rna_name)
                output["model_name"].append(method)
                output["mae"].append(maes)
        df = pd.DataFrame(output)
        return df

    def get_mae_for_preds(self, name="dnabert"):
        """
        Get the MAE for the predictions
        """
        df_mae = self.get_mae()
        df_mae.groupby(["rna_name", "model_name"]).mean()
        top1, top5 = self.get_best_model(df_mae, name)
        return top1, top5

    def get_best_model(self, df: pd.DataFrame, name: str = "dnabert"):
        new_output = {"model_name": [], "rna_name": []}
        count_dnabert_top_5 = 0
        for rna_name in df["rna_name"].unique():
            current_df = df[df["rna_name"] == rna_name]
            current_df = current_df.sort_values("mae")
            name = current_df.iloc[0]["model_name"].split("-")[0]
            if name in current_df.iloc[:5]["model_name"].values:
                count_dnabert_top_5 += 1
            new_output["model_name"].append(name)
            new_output["rna_name"].append(rna_name)
        df = pd.DataFrame(new_output)
        count_dnabert_top_1 = len(df[df["model_name"] == name])
        return count_dnabert_top_1, count_dnabert_top_5

    @staticmethod
    def get_count_all_datasets(save_path: str):
        prefix = os.path.join("data", "angles_decoys")
        all_dfs = {"dataset": [], "top1": [], "top5": []}
        for name in ["pm_decoys", "rna_puzzles_decoys", "near_native"]:
            params = {
                "native_json_path": os.path.join(prefix, "native", f"{name}.json"),
                "preds_json_path": os.path.join(prefix, "preds", name),
            }
            viz_decoys = Decoys(**params)
            top1, top5 = viz_decoys.get_mae_for_preds()
            all_dfs["dataset"].append(name)
            all_dfs["top1"].append(top1)
            all_dfs["top5"].append(top5)
        df = pd.DataFrame(all_dfs)
        df.to_latex(save_path, index=False)
