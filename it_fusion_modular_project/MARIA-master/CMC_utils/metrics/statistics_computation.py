import os
import re
import logging
import pandas as pd
from typing import List, Union
from omegaconf import DictConfig
from CMC_utils import save_load
from CMC_utils.paths import get_files_with_extension
from .performance_computation import *
from .late_fusion_computation import *


log = logging.getLogger(__name__)

__all__ = ["compute_performance", "compute_missing_performance", "compute_late_fusion_performance", "compute_late_fusion_missing_performance"]


def compute_late_fusion_missing_performance(classes: List[str], prediction_path: str, results_path: str, task: str, metrics: Union[dict, DictConfig], missing_percentages: List[float], datasets_names: List[str], late_fusion_approaches: List[str], decimals: int = 2) -> None:
    missing_percentages = [int(miss_perc * 100) for miss_perc in missing_percentages]

    for train_missing_percentage in missing_percentages:
        preds_path = os.path.join( prediction_path, str(train_missing_percentage) )
        res_path = os.path.join( results_path, str(train_missing_percentage) )

        compute_late_fusion_performance(classes=classes, prediction_path=preds_path, results_path=res_path, metrics=metrics, datasets_names=datasets_names, late_fusion_approaches=late_fusion_approaches, task=task, decimals=decimals)

        for test_missing_percentage in missing_percentages:
            test_preds_path = os.path.join( preds_path, str(test_missing_percentage) )
            test_res_path = os.path.join(res_path, str(test_missing_percentage))

            compute_late_fusion_performance(classes=classes, prediction_path=test_preds_path, results_path=test_res_path, metrics=metrics, datasets_names=datasets_names, late_fusion_approaches=late_fusion_approaches, task=task, decimals=decimals)


def compute_late_fusion_performance(classes: List[str], prediction_path: str, results_path: str, task: str, metrics: Union[dict, DictConfig], datasets_names: List[str], late_fusion_approaches: List[str], decimals: int = 2) -> None:
    datasets_folders = [os.path.join(prediction_path, dataset_name) for dataset_name in datasets_names]

    datasets_prediction_files = []
    for dataset_folder in datasets_folders:
        prediction_files = get_files_with_extension(dataset_folder, "csv")
        prediction_files = sorted(prediction_files, key=lambda file_path: int(re.findall(r"\d+_(\d+)_\w+\.csv\Z", file_path)[0]))
        prediction_files = sorted(prediction_files, key=lambda file_path: int(re.findall(r"(\d+)_\d+_\w+\.csv\Z", file_path)[0]))
        datasets_prediction_files.append(prediction_files)

    for fset in ["train", "val", "test"]:
        set_files = [[prediction_file for prediction_file in dataset_prediction_files if prediction_file.endswith(f"{fset}.csv")] for dataset_prediction_files in datasets_prediction_files]

        performance, performance_balanced = {late_fusion_approach: pd.DataFrame() for late_fusion_approach in late_fusion_approaches}, {late_fusion_approach: pd.DataFrame() for late_fusion_approach in late_fusion_approaches}
        for files in zip(*set_files):
            assert all(os.path.basename(file) == os.path.basename(files[0]) for file in files), "Files are not aligned"

            test_fold, val_fold = [int(fold_num) for fold_num in re.findall(r"(\d+)_(\d+)_\w+\.csv\Z", files[0])[0]]

            decision_profile = get_decision_profile(files)
            for late_fusion_approach in late_fusion_approaches:
                fold_preds = apply_late_fusion_approach(decision_profile, late_fusion_approach=late_fusion_approach, task=task, classes=classes)

                if task in ("classification"):
                    fold_performance = metrics_computation_df(fold_preds, task=task, metrics=metrics, classes=classes, use_weights=False, verbose=0).mul(100).round(2).reset_index().assign(set=fset, test_fold=test_fold, val_fold=val_fold)
                    fold_performance_balanced = metrics_computation_df(fold_preds, task=task, metrics=metrics, classes=classes, use_weights=True, verbose=0).mul(100).round(2).reset_index().assign(set=fset, test_fold=test_fold, val_fold=val_fold)
                else:
                    fold_performance = metrics_computation_df(fold_preds, task=task, metrics=metrics, classes=classes, use_weights=False, verbose=0).round(decimals).assign(set=fset, test_fold=test_fold, val_fold=val_fold)
                    fold_performance_balanced = metrics_computation_df(fold_preds, task=task, metrics=metrics, classes=classes, use_weights=True, verbose=0).round(decimals).assign(set=fset, test_fold=test_fold, val_fold=val_fold)

                performance[late_fusion_approach] = pd.concat([performance[late_fusion_approach], fold_performance], axis=0, ignore_index=True)
                performance_balanced[late_fusion_approach] = pd.concat([performance_balanced[late_fusion_approach], fold_performance_balanced], axis=0, ignore_index=True)

        for late_fusion_approach in late_fusion_approaches:
            perf = performance[late_fusion_approach]
            if not perf.empty:
                unbalanced_path = os.path.join(results_path, "unbalanced", fset, late_fusion_approach)
                if not os.path.exists(unbalanced_path):
                    os.makedirs(unbalanced_path)
                compute_performance_statistics(perf, unbalanced_path, task, decimals=decimals)
            perf_bal = performance_balanced[late_fusion_approach]
            if not perf_bal.empty and task in ("classification"):
                balanced_path = os.path.join(results_path, "balanced", fset, late_fusion_approach)
                if not os.path.exists(balanced_path):
                    os.makedirs(balanced_path)
                compute_performance_statistics(perf_bal, balanced_path, task, decimals=decimals)


def compute_missing_performance(classes: List[str], prediction_path: str, results_path: str, task: str, metrics: Union[dict, DictConfig], missing_percentages: List[float], **kwargs) -> None:
    """
    Compute the performance for different missing percentages
    Parameters
    ----------
    classes : List[str]
    prediction_path : str
    results_path : str
    metrics : Union[dict, DictConfig]
    missing_percentages : List[float]

    Returns
    -------
    None
    """
    missing_percentages = [int(miss_perc * 100) for miss_perc in missing_percentages]

    for train_missing_percentage in missing_percentages:
        preds_path = os.path.join( prediction_path, str(train_missing_percentage) )
        res_path = os.path.join( results_path, str(train_missing_percentage) )

        compute_performance(classes=classes, prediction_path=preds_path, results_path=res_path, task=task, metrics=metrics, **kwargs)

        for test_missing_percentage in missing_percentages:
            test_preds_path = os.path.join( preds_path, str(test_missing_percentage) )
            test_res_path = os.path.join(res_path, str(test_missing_percentage))
            compute_performance(classes=classes, prediction_path=test_preds_path, results_path=test_res_path, task=task, metrics=metrics, **kwargs)


def compute_performance(classes: List[str], prediction_path: str, results_path: str, task: str, metrics: Union[dict, DictConfig], decimals: int = 2, **_) -> None:
    """
    Compute the performance for a given set of predictions
    Parameters
    ----------
    classes : List[str]
    prediction_path : str
    results_path : str
    metrics : Union[dict, DictConfig]

    Returns
    -------
    None
    """
    prediction_files = get_files_with_extension(prediction_path, "csv")
    prediction_files = sorted(prediction_files, key=lambda file_path: int(re.findall(r"\d+_(\d+)_\w+\.csv\Z", file_path)[0]))
    prediction_files = sorted(prediction_files, key=lambda file_path: int(re.findall(r"(\d+)_\d+_\w+\.csv\Z", file_path)[0]))

    for fset in ["train", "val", "test"]:
        set_files = [file for file in prediction_files if file.endswith(f"{fset}.csv")]

        performance, performance_balanced = pd.DataFrame(), pd.DataFrame()

        for file in set_files:
            test_fold, val_fold = [ int(fold_num) for fold_num in re.findall(r"(\d+)_(\d+)_\w+\.csv\Z", file)[0] ]

            fold_preds = save_load.load_params_table(file).set_index("ID")
            if task in ("classification", "survival_analysis"):
                fold_performance = metrics_computation_df(fold_preds, task=task, metrics=metrics, classes=classes, use_weights=False, verbose=0).mul(100).round(decimals).reset_index().assign(set=fset, test_fold=test_fold, val_fold=val_fold)
                fold_performance_balanced = metrics_computation_df(fold_preds, task=task, metrics=metrics, classes=classes, use_weights=True, verbose=0).mul(100).round(decimals).reset_index().assign(set=fset, test_fold=test_fold, val_fold=val_fold)
            else:
                fold_performance = metrics_computation_df(fold_preds, task=task, metrics=metrics, classes=classes, use_weights=False, verbose=0).round(decimals).assign(set=fset, test_fold=test_fold, val_fold=val_fold)
                fold_performance_balanced = metrics_computation_df(fold_preds, task=task, metrics=metrics, classes=classes, use_weights=True, verbose=0).round(decimals).assign(set=fset, test_fold=test_fold, val_fold=val_fold)

            performance = pd.concat([ performance, fold_performance ], axis=0, ignore_index=True )
            performance_balanced = pd.concat([ performance_balanced, fold_performance_balanced ], axis=0, ignore_index=True )

        if not performance.empty:
            unbalanced_path = os.path.join(results_path, "unbalanced", fset)
            if not os.path.exists(unbalanced_path):
                os.makedirs(unbalanced_path)
            compute_performance_statistics(performance, unbalanced_path, task, decimals=decimals)

        if not performance_balanced.empty and task in ("classification"):
            balanced_path = os.path.join(results_path, "balanced", fset)
            if not os.path.exists(balanced_path):
                os.makedirs(balanced_path)
            compute_performance_statistics(performance_balanced, balanced_path, task, decimals=decimals)


def compute_performance_statistics(performance, path, task, decimals: int = 2) -> None:
    """
    Compute the performance statistics
    Parameters
    ----------
    performance : pd.DataFrame
    path : str

    Returns
    -------
   None
    """
    save_load.save_table(performance, f"all_test_performance.csv", path, index=False)
    if task in ("classification", "survival_analysis"):
        mean_performance = performance.drop(["test_fold", "val_fold"], axis=1).groupby( by=["set", "class"] ).agg( [ "mean", "std", "min", "max" ] ).round(decimals)
        save_load.save_table( mean_performance.reset_index(), f"classes_average_performance.csv", path, index=False )

    average_performance_drop = ["test_fold", "val_fold", "class"] if task in ("classification", "survival_analysis") else ["test_fold", "val_fold"]
    average_performance = performance.drop(average_performance_drop, axis=1).groupby(by=["set"]).agg(["mean", "std", "min", "max"]).round(decimals)
    save_load.save_table( average_performance.reset_index(), f"set_average_performance.csv", path, index=False )
    log.info("Average performance computed")


if __name__ == "__main__":
    pass
