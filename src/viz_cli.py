import os

from src.helper.all_decoys_correlation import AllDecoysCorrelation
from src.helper.decoys_correlation import VizDecoysCorrelation
from src.helper.score_comparer import ScoreComparer
from src.helper.stats import Stats
import argparse


class VizCLI:
    def __init__(self, method: str):
        self.method = method
        self.conversion = {
            "data": self.viz_polar_distribution,
            "stats": self.get_dataset_stats,
            "decoys": self.get_all_correlation,
            "model": self.get_plot_models,
        }

    def viz(self):
        self.conversion[self.method]()

    @staticmethod
    def viz_polar_distribution():
        """
        Visualize the polar distribution of the angles in the different sets.
        """
        path_to_save = os.path.join("img", "polar_distribution.png")
        Stats.plot_distribution_polar_data(path_to_save)

    @staticmethod
    def get_dataset_stats():
        """
        Get the stats of the datasets
        """
        Stats.get_data_len()

    @staticmethod
    def get_all_correlation():
        # Test Set I
        VizDecoysCorrelation.compute_correlation_test_set(1)
        # Test Set II
        VizDecoysCorrelation.compute_correlation_test_set(2)
        # Test Set III
        VizDecoysCorrelation.compute_correlation_test_set(3)
        AllDecoysCorrelation.get_all_energies()
        AllDecoysCorrelation.get_all_metrics()

    @staticmethod
    def get_plot_models():
        ScoreComparer.get_df_mae_all_test()
        ScoreComparer.get_len_plot()
        ScoreComparer.get_table_results()

    @staticmethod
    def get_arguments():
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--method",
            type=str,
            default="data",
            choices=["data", "stats", "decoys", "model"],
            help="Method to visualize",
        )
        args = parser.parse_args()
        return args


if __name__ == "__main__":
    args = VizCLI.get_arguments()
    viz_cli = VizCLI(**vars(args))
    viz_cli.viz()
