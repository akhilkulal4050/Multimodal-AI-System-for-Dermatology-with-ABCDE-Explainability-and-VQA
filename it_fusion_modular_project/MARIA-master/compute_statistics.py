import sys
sys.path.append("CMC_utils")

import os
import re
import scipy
import numpy as np
import pandas as pd
from tqdm import tqdm
tqdm.pandas()
from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True, nb_workers=4)
from CMC_utils import save_load, paths
from CMC_utils.metrics.late_fusion_computation import get_decision_profile, apply_late_fusion_approach


def load_single_experiment_predictions(path: str, tr_rates: list, te_rates: list, late_fusion: bool = False):
    all_predictions = pd.DataFrame()
    for tr_rate in tr_rates:
        for te_rate in te_rates:
            exp_path = os.path.join(path, str(tr_rate), str(te_rate))
            if not late_fusion:
                files = paths.get_files_with_extension(exp_path, "csv")
                files = [file for file in files if file.endswith("_test.csv")]
                files = pd.DataFrame( {"file": files} )
                files = files.assign(train_fold=files.file.apply(lambda file_name: int( re.findall(r"(\d+)_\d+_test\.csv\Z", file_name)[0] )))
                files = files.assign(val_fold=files.file.apply(lambda file_name: int( re.findall(r"\d+_(\d+)_test\.csv\Z", file_name)[0] )))
                files = files.assign(train_fraction=tr_rate)
                files = files.assign(test_fraction=te_rate)
                files = files.sort_values(by=["train_fold", "val_fold", "train_fraction", "test_fraction"]).set_index(["train_fold", "val_fold", "train_fraction", "test_fraction"])

                for (train_fold, val_fold, train_fraction, test_fraction), file in files.iterrows():
                    predictions = save_load.load_table(file.iloc[0], header=0).assign(train_fold=train_fold, val_fold=val_fold, train_fraction=train_fraction, test_fraction=test_fraction)
                    all_predictions = pd.concat([all_predictions, predictions], axis=0, ignore_index=True)
            else:
                datasets_folders = os.listdir(exp_path)
                datasets_prediction_files = []
                for dataset_folder in datasets_folders:
                    prediction_files = paths.get_files_with_extension(os.path.join(exp_path, dataset_folder), "csv")
                    prediction_files = sorted(prediction_files, key=lambda file_path: int(re.findall(r"\d+_(\d+)_\w+\.csv\Z", file_path)[0]))
                    prediction_files = sorted(prediction_files, key=lambda file_path: int(re.findall(r"(\d+)_\d+_\w+\.csv\Z", file_path)[0]))
                    datasets_prediction_files.append(prediction_files)

                for files in zip(*datasets_prediction_files):
                    decision_profile = get_decision_profile(files)
                    train_fold, val_fold = [int(fold_num) for fold_num in re.findall(r"(\d+)_(\d+)_\w+\.csv\Z", files[0])[0]]
                    classes = decision_profile.label.unique().tolist()

                    fold_preds = apply_late_fusion_approach(decision_profile, late_fusion_approach="mean", task="classification", classes=classes)
                    fold_preds = fold_preds.assign(train_fold=train_fold, val_fold=val_fold, train_fraction=tr_rate, test_fraction=te_rate).reset_index()
                    all_predictions = pd.concat([all_predictions, fold_preds], axis=0, ignore_index=True)

    return all_predictions


def compute_single_MARIA_vs_model_strategy_imputer(competitor_path: str, tr_rates: list, te_rates: list, MARIA_predictions: pd.DataFrame, separate_folds: bool = False):
    competitor_path = competitor_path.iloc[0]
    correctly_classified = lambda sample: int(sample.label == sample.prediction)

    model = competitor_path.split("/")[-1].split("_")[0]
    strategy = re.findall(r"(early|joint|late)", competitor_path)[0]
    imputer = re.findall(r"(noimputation|simple|knn|iterative)", competitor_path)[0]

    competitor_predictions = load_single_experiment_predictions(competitor_path, tr_rates, te_rates, late_fusion=strategy == "late")
    competitor_predictions = competitor_predictions.assign( correctly_classified = competitor_predictions.apply(correctly_classified, axis=1) )

    MARIA_better_results_counter = 0
    MARIA_worse_results_counter = 0
    for train_fraction in competitor_predictions.train_fraction.unique():
        for test_fraction in competitor_predictions.test_fraction.unique():
            MARIA_preds_map = pd.concat([MARIA_predictions.train_fraction == train_fraction, MARIA_predictions.test_fraction == test_fraction], axis=1 ).all(axis=1)

            MARIA_preds = MARIA_predictions.loc[MARIA_preds_map, ["ID", "train_fold", "correctly_classified"]].set_index("ID")

            competitor_preds_map = pd.concat([competitor_predictions.train_fraction == train_fraction, competitor_predictions.test_fraction == test_fraction], axis=1 ).all(axis=1)
            competitor_preds = competitor_predictions.loc[competitor_preds_map, ["ID", "train_fold", "correctly_classified"]].set_index("ID")

            if not separate_folds:
                MARIA_preds.train_fold = -1
                competitor_preds.train_fold = -1

            for fold in competitor_preds.train_fold.unique():
                MARIA = MARIA_preds.loc[MARIA_preds.train_fold == fold, "correctly_classified"]
                competitor = competitor_preds.loc[competitor_preds.train_fold == fold, "correctly_classified"]
                competitor = competitor.loc[MARIA.index]

                [_, p_value] = scipy.stats.wilcoxon(MARIA, y=competitor, zero_method='zsplit', correction=False, alternative='greater', method='auto')
                if p_value < 0.05:
                    MARIA_better_results_counter += 1

                [_, p_value2] = scipy.stats.wilcoxon(MARIA, y=competitor, zero_method='zsplit', correction=False, alternative='less', method='auto')
                if p_value2 < 0.05:
                    MARIA_worse_results_counter += 1

    fold_numbers = competitor_predictions.train_fold.nunique() if separate_folds else 1
    num_experiment = competitor_predictions.train_fraction.nunique() * competitor_predictions.test_fraction.nunique() * fold_numbers
    MARIA_better_results_perc = np.round(100 * MARIA_better_results_counter / num_experiment, 2)
    MARIA_worse_results_perc = np.round(100 * MARIA_worse_results_counter / num_experiment, 2)
    return model, strategy, imputer, MARIA_better_results_perc, MARIA_worse_results_perc


def compute_MARIA_vs_model_strategy_imputer(maria_path: str, *competitors_paths: str, missing_scenario: str, task_name: str, separate_folds: bool = False):
    if missing_scenario == "all":
        if task_name in ("ADNI_prognosis_m36", "AI4Covid_death", "AI4Covid_prognosis"):
            tr_rates = [0, 30, 50, 75]
        else:
            tr_rates = [0, 50, 75]
    else:
        tr_rates = [0, 10, 30, 50, 75]
    te_rates = tr_rates

    correctly_classified = lambda sample: int(sample.label == sample.prediction)
    MARIA_predictions = load_single_experiment_predictions(maria_path, tr_rates, te_rates)
    MARIA_predictions = MARIA_predictions.assign(correctly_classified = MARIA_predictions.apply(correctly_classified, axis=1))

    competitors_paths = pd.DataFrame(competitors_paths)
    results = competitors_paths.parallel_apply(compute_single_MARIA_vs_model_strategy_imputer, tr_rates=tr_rates, te_rates=te_rates, MARIA_predictions=MARIA_predictions.copy(), separate_folds=separate_folds, axis=1, result_type="expand").rename(columns={0: "model", 1: "strategy", 2: "imputer", 3: "MARIA_better_percentage",  4: "MARIA_worse_percentage"})

    results = results.set_index(["strategy", "model", "imputer"]).unstack()

    better_results = results.MARIA_better_percentage.fillna('').astype(str)
    worse_results = results.MARIA_worse_percentage.fillna('').astype(str)

    result_df = better_results.add('-' + worse_results)

    return result_df

def filter_paths(path: str, imputers: list, missing_scenario: str, completely_remove_naim: bool = False):
    select_path = lambda p, substr: [os.path.join(p, dirname) for dirname in os.listdir(p) if os.path.isdir(os.path.join(p, dirname)) and re.search(substr, dirname) is not None]
    selected_paths = [select_path(path, missing_scenario), []]

    for substring in imputers + ["noimputation"]:
        selected_paths[1] += select_path(path, substring)

    if completely_remove_naim:
        selected_paths[1] = [p for p in selected_paths[1] if re.search("naim", p) is None]
    else:
        selected_paths[1] = [p for p in selected_paths[1] if re.search("naimmean", p) is None and re.search("naimclstoken", p) is None]
    selected_paths = set(selected_paths[0]).intersection(set(selected_paths[1]))
    return list(selected_paths)

def compute_MARIA_vs_model_strategy_imputer_combinations(*tasks_paths: str, output_path: str, missing_scenario: str, imputers: list, separate_folds: bool = False):
    datasets_tables = dict()
    for task_path in tasks_paths:
        early_path = os.path.join(task_path + "_42_multimodal_early_fusion_classification_with_missing_generation", "stratifiedkfold_holdout", "predictions")
        joint_path = os.path.join(task_path + "_42_multimodal_joint_fusion_classification_with_missing_generation", "stratifiedkfold_holdout", "predictions")
        late_path = os.path.join(task_path + "_42_multimodal_late_fusion_classification_with_missing_generation", "stratifiedkfold_holdout", "predictions")

        early_paths = filter_paths(early_path, imputers, missing_scenario, completely_remove_naim=True)
        joint_paths = filter_paths(joint_path, imputers, missing_scenario)
        late_paths = filter_paths(late_path, imputers, missing_scenario, completely_remove_naim=True)

        maria_path = [path for path in joint_paths if re.search("naimcat", path) is not None][0]
        competitors_paths = [path for path in joint_paths if re.search("naimcat", path) is None]
        competitors_paths = early_paths + competitors_paths + late_paths

        task_name = task_path.split("/")[-1]

        task_results = compute_MARIA_vs_model_strategy_imputer(maria_path, *competitors_paths, missing_scenario=missing_scenario, task_name=task_name, separate_folds=separate_folds)
        datasets_tables[task_name] = task_results

    return datasets_tables


if __name__ == "__main__":
    base_path = "/Volumes/Aaron SSD/UCBM/Projects"
    # base_path = "/Users/camillocaruso/LocalDocuments/code_outputs"

    tasks_paths = [
        "multiNAIM_ADNI/ADNI_diagnosis_binary",
        "multiNAIM_ADNI/ADNI_diagnosis_multiclass",
        "multiNAIM_ADNI/ADNI_prognosis_m12",
        "multiNAIM_ADNI/ADNI_prognosis_m24",
        "multiNAIM_ADNI/ADNI_prognosis_m36",
        "multiNAIM_ADNI/ADNI_prognosis_m48",
        "multiNAIM_AI4Covid/AI4Covid_death",
        "multiNAIM_AI4Covid/AI4Covid_prognosis"
    ]

    """tasks_paths = [
        "multiNAIM_ADNI/ADNI_diagnosis_binary",
        "multiNAIM_ADNI/ADNI_prognosis_m48"
    ]"""

    tasks_paths = [os.path.join(base_path, path) for path in tasks_paths]

    output_path = "/Users/camillocaruso/Downloads/plots"

    imputers = ["knn"]

    ml_models = ["AdaBoost", "DecisionTree", "HistGradientBoost", "RandomForest", "SVM", "XGBoost"]
    ml_models_wo_imputers = ["DecisionTree", "HistGradientBoost", "RandomForest", "XGBoost"]
    dl_models = ["MLP", "TabNet", "TabTransformer", "FTTransformer"]

    for missing_scenario in ["all", "modalities"]:
        print(f"Computing MARIA vs model strategy imputer combinations for missing scenario: {missing_scenario}")

        datasets_results_dict = compute_MARIA_vs_model_strategy_imputer_combinations(*tasks_paths, output_path=output_path, missing_scenario=missing_scenario, imputers=imputers, separate_folds=False)
        datasets_results = pd.concat([datasets_results_dict[dataset].stack().rename(dataset) for dataset in datasets_results_dict], axis=1).reset_index()

        models_map = {"adaboost": "AdaBoost", "dt": "DecisionTree", "histgradientboostingtree": "HistGradientBoost", "rf": "RandomForest", "svm": "SVM", "xgboost": "XGBoost"}
        datasets_results.model = datasets_results.model.map(lambda x: x if x not in models_map.keys() else models_map[x])
        datasets_results.imputer = datasets_results.imputer.map({"knn": "with", "noimputation": "without"})
        datasets_results = datasets_results.set_index(["strategy", "model", "imputer"])

        res_splitted = pd.DataFrame()
        for col in datasets_results.columns:
            single_res_splitted = datasets_results[col].str.split('-', expand=True).replace("", np.nan).astype(float).dropna(how="all", axis=0)
            single_res_splitted.columns = pd.MultiIndex.from_product([[col], ["\% Win", "\% Loss"]])
            res_splitted = pd.concat([res_splitted, single_res_splitted], axis=1)

        # for strategy in ["early", "joint", "late"]:
        #     res_splitted.loc[(strategy, "Mean", "Mean"), :] = res_splitted.loc[strategy].mean(axis=0)

        for strategy, models_dict, imputer in zip(["early", "early", "early", "joint", "late", "late", "late"], [{"ML": ml_models}, {"ML": ml_models_wo_imputers}, {"DL": dl_models}, {"DL": dl_models}, {"ML": ml_models}, {"ML": ml_models_wo_imputers}, {"DL": dl_models}], ["with", "without", "with", "with", "with", "without", "with"]):
            models_type, models = list(models_dict.items())[0]
            res_splitted.loc[(strategy, models_type, imputer), :] = res_splitted.loc[(strategy, models, imputer)].mean(axis=0)

        for scenario in ["\% Win", "\% Loss"]:
            mask = res_splitted.columns.get_level_values(1) == scenario
            selected_columns = res_splitted.loc[:, mask]
            res_splitted[("Mean", scenario)] = np.nanmean(selected_columns, axis=1)

        res_splitted.to_latex(os.path.join(output_path, f"MARIA_vs_model_strategy_imputer_combinations_{missing_scenario}.txt"),
                                 float_format="%.2f",
                                 column_format="lllcccc",
                                 escape=False,
                                 index=True,
                                 multirow=True,
                                 multicolumn=True,
                                 caption=f"MARIA vs model strategy imputer combinations for missing scenario: {missing_scenario}",
                                 label=f"tab:MARIA_vs_model_strategy_imputer_combinations_{missing_scenario}")

