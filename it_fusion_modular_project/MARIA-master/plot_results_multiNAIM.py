import sys

sys.path.append("CMC_utils")

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from CMC_utils.plots import load_missing_results

from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True, nb_workers=4)

sns.set_style("whitegrid")
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Cambria']})
rc('text', usetex=True)


def plot_performance(data: pd.DataFrame, metrics: dict, filename: str, output_path: str, extension: str = "png"):
    data.loc[:, "style"] = data.fusion_strategy.str.capitalize().str.cat(data.model, sep=" ")
    data.loc[data["style"] == "Joint NAIM", "style"] = "MARIA"
    data.loc[:, "style"] = data["style"].str.replace("Joint", "Intermediate")

    data_grouped = data.groupby('db')

    rowlength = np.ceil(data_grouped.ngroups / 2).astype(int)

    fig, axs = plt.subplots(figsize=(20, 15), nrows=2, ncols=rowlength, gridspec_kw=dict(hspace=0.3, wspace= 0.3))

    targets = zip(data_grouped.groups.keys(), axs.flatten())
    for i, (key, ax) in enumerate(targets):
        percentages = sorted(data_grouped.get_group(key).test_percentage.unique())
        percentages_str = [f"{perc}\%" for perc in percentages[1:]]
        percentages_str = ["$\mathbf{\Omega}$"] + [r"\textbf{" + perc + "}" for perc in percentages_str]

        ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        ax.set_yticks(np.arange(len(percentages))/len(percentages) + (1/(2*len(percentages))))
        ax.set_yticklabels(percentages_str)
        ax.tick_params(axis='x', labelsize=13)
        ax.tick_params(axis='y', labelsize=13)
        plt.setp(ax.get_yticklabels(), rotation=0, ha='left', va='center')
        ax.set_ylabel('')

        ax.grid(False)

        for label in ax.get_yticklabels():
            label.set_x(-0.16)

        y_min, y_max = np.floor(data_grouped.get_group(key)[metrics[key]].min()).astype(int), np.ceil(data_grouped.get_group(key)[metrics[key]].max()).astype(int)

        db_data = data_grouped.get_group(key).groupby("train_percentage")

        for j, perc in enumerate(percentages):
            inset_ax = ax.inset_axes([0, j * (1 / len(percentages)), 1, 1 / len(percentages)])
            perc_db_data = db_data.get_group(perc)
            perc_db_data.loc[:, "x"] = perc_db_data["test_percentage"].map(lambda x: percentages.index(x)/len(percentages))

            sns.lineplot(data=perc_db_data, x="x", y=metrics[key], hue="style", markers=True, dashes=False, ax=inset_ax, errorbar=None, linewidth=1.5)

            inset_ax.spines['top'].set_linewidth(1.5)
            inset_ax.spines['right'].set_linewidth(1.5)
            inset_ax.spines['bottom'].set_linewidth(1.5)
            inset_ax.spines['left'].set_linewidth(1.5)

            for spine in inset_ax.spines.values():
                spine.set_color('black')

            if j != 0:
                inset_ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False )

            inset_ax.tick_params(axis='y', which='both', left=True, labelleft=True, labelsize=13)
            inset_ax.set_yticks([y_min, y_max])

            inset_ax.set_ylabel('')
            inset_ax.set_xlabel('')

            if j == len(percentages) - 1:
                inset_ax.set_title(key.replace("_", " ") + f"\n {metrics[key].upper()}", fontsize=24, pad=15)

            inset_ax.set_xticks(np.arange(len(percentages))/len(percentages))

            if j == 0:
                inset_ax.tick_params(axis='x', which='both', bottom=True, top=True, labelbottom=True, labelsize=13)
                inset_ax.set_xticklabels(percentages_str)

            margin = 0.13 * (y_max - y_min)
            inset_ax.set_ylim(y_min - margin, y_max + margin)

            handles, labels = inset_ax.get_legend_handles_labels()

            for handle in handles:
                handle.set_linewidth(3)

            fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.03), fontsize=16, ncols=len(labels))

            inset_ax.get_legend().remove()

    if len(axs.flatten())-1 > i:
        for j in range(i+1, len(axs.flatten())):
            axs.flatten()[j].axis('off')

    plt.suptitle(filename.replace("_", " "), fontsize=24, fontweight='extra bold')

    plt.savefig(os.path.join(output_path, f'{filename}.{extension}'), format=extension)
    plt.close()


def plot_missing_results(result_paths, plots_output_path, imputers, extension="png"):
    if not os.path.exists(plots_output_path):
        os.makedirs(plots_output_path)

    result_paths = pd.Series([os.path.join(path, "stratifiedkfold_holdout") for path in result_paths])
    mean_results_path = os.path.join(plots_output_path, "mean_results.csv")
    if not os.path.exists(mean_results_path):
        all_results = pd.concat( result_paths.parallel_apply(lambda x: load_missing_results(x, result_type="set", separate_experiments=False, verbose=False)).values, axis=0)

        mean_cols = [col for col in all_results.columns if col.endswith("_mean")]
        mean_results = all_results[mean_cols].rename(columns={col: col.replace("_mean", "") for col in mean_cols})
        del mean_cols, all_results

        mean_results = mean_results.reset_index()
        mean_results.to_csv(mean_results_path, index=False)
    else:
        mean_results = pd.read_csv(mean_results_path, header=0)
    del result_paths, mean_results_path

    mean_results = mean_results.loc[~mean_results.model.isin(["naimmean", "naimclstoken"])].reset_index(drop=True)

    DL_models = {"naimcat": "NAIM", "MLP": "MLP", "FTTransformer": "FTTransformer", "TabTransformer": "TABTransformer", "TabNet": "TabNet"}
    ML_models = {"naimcat": "NAIM", "xgboost": "XGBoost", "rf": "RandomForest", "dt": "DecisionTree", "adaboost": "AdaBoost", "histgradientboostingtree": "HistGradientBoost", "svm": "SVM"}

    mean_results["model"] = mean_results.model.apply(lambda x: DL_models.get(x, ML_models.get(x, x)))

    mean_results = mean_results.drop("experiment", axis=1)
    mean_results = mean_results.loc[mean_results.imputer.isin(imputers + ["noimputation"])].reset_index(drop=True)
    del imputers

    mean_results.loc[:, "fusion_strategy"] = mean_results.fusion_strategy.where(mean_results.fusion_strategy != "latemean", "late")
    mean_results = mean_results.loc[mean_results.fusion_strategy.isin(["early", "joint", "late"])].reset_index(drop=True)
    mean_results = mean_results.loc[mean_results.missing_strategy.isin(["all", "modalities"])].reset_index(drop=True)

    mean_results.loc[:, "db"] = mean_results.db.str.replace("AI4Covid", "AIforCOVID")

    naim_map = mean_results.model.str.lower().str.contains("naim")
    all_naim_results = mean_results.loc[naim_map]
    competitors_results = mean_results.loc[~naim_map]
    DL_competitors_results = competitors_results.loc[competitors_results.model.isin(DL_models.values())].reset_index(drop=True)
    ML_competitors_results = competitors_results.loc[competitors_results.model.isin(ML_models.values())].reset_index(drop=True)
    NAIM_competitors_results = all_naim_results.loc[all_naim_results.fusion_strategy.isin(["early", "late"])].reset_index(drop=True)
    NAIM_results = all_naim_results.loc[all_naim_results.fusion_strategy == "joint"].reset_index(drop=True)
    NAIM_results_all = NAIM_results.loc[NAIM_results.missing_strategy == "all"].drop("missing_strategy", axis=1).reset_index(drop=True)
    NAIM_results_modalities = NAIM_results.loc[NAIM_results.missing_strategy == "modalities"].drop("missing_strategy", axis=1).reset_index(drop=True)
    del naim_map, all_naim_results, competitors_results, NAIM_results, mean_results

    all_original_missing_percentages = {"ADNI_diagnosis_binary": 49, "ADNI_diagnosis_multiclass": 49,
                                        "ADNI_prognosis_m12": 36, "ADNI_prognosis_m24": 35, "AIforCOVID_death": 23,
                                        "AIforCOVID_prognosis": 23, "ADNI_prognosis_m36": 29, "ADNI_prognosis_m48": 37}

    dbs_metrics = dict(ADNI_diagnosis_binary="auc", ADNI_diagnosis_multiclass="auc", ADNI_prognosis_m12="mcc", ADNI_prognosis_m24="mcc", AIforCOVID_death="mcc", AIforCOVID_prognosis="auc", ADNI_prognosis_m36="mcc", ADNI_prognosis_m48="mcc")

    train_percentage_rule = lambda row, info: row.train_percentage > info[row.db] or row.train_percentage == 0
    test_percentage_rule = lambda row, info: row.test_percentage > info[row.db] or row.test_percentage == 0

    NAIM_results_all = NAIM_results_all.loc[NAIM_results_all.apply(lambda x: train_percentage_rule(x, all_original_missing_percentages), axis=1)].reset_index(drop=True)
    NAIM_results_all = NAIM_results_all.loc[NAIM_results_all.apply(lambda x: test_percentage_rule(x, all_original_missing_percentages), axis=1)].reset_index(drop=True)

    ####################################################################################################################

    # Deep Learning

    DL_competitors_all = DL_competitors_results.loc[DL_competitors_results.missing_strategy == "all"].drop("missing_strategy", axis=1)
    DL_competitors_all = DL_competitors_all.loc[DL_competitors_all.apply(lambda x: train_percentage_rule(x, all_original_missing_percentages), axis=1)].reset_index(drop=True)
    DL_competitors_all = DL_competitors_all.loc[DL_competitors_all.apply(lambda x: test_percentage_rule(x, all_original_missing_percentages), axis=1)].reset_index(drop=True)
    A = DL_competitors_all.copy()
    DL_competitors_all = DL_competitors_all.drop("model", axis=1).groupby(by=["db", "imputer", "fusion_strategy", "train_percentage", "test_percentage"]).mean().round(2).reset_index()
    DL_competitors_all.loc[:, "model"] = "DL"
    NAIM_vs_DL_all = pd.concat([NAIM_results_all, DL_competitors_all], axis=0).reset_index(drop=True)
    del DL_competitors_all

    DL_competitors_modalities = DL_competitors_results.loc[DL_competitors_results.missing_strategy == "modalities"].drop("missing_strategy", axis=1)
    B = DL_competitors_modalities.copy()
    DL_competitors_modalities = DL_competitors_modalities.drop("model", axis=1).groupby(by=["db", "imputer", "fusion_strategy", "train_percentage", "test_percentage"]).mean().round(2).reset_index()
    DL_competitors_modalities.loc[:, "model"] = "DL"
    NAIM_vs_DL_modalities = pd.concat([NAIM_results_modalities, DL_competitors_modalities], axis=0).reset_index(drop=True)
    del DL_competitors_modalities, DL_competitors_results

    ####################################################################################################################

    # Machine Learning

    ML_competitors_all = ML_competitors_results.loc[ML_competitors_results.missing_strategy == "all"].drop("missing_strategy", axis=1)
    ML_competitors_all = ML_competitors_all.loc[ML_competitors_all.apply(lambda x: train_percentage_rule(x, all_original_missing_percentages), axis=1)].reset_index(drop=True)
    ML_competitors_all = ML_competitors_all.loc[ML_competitors_all.apply(lambda x: test_percentage_rule(x, all_original_missing_percentages), axis=1)].reset_index(drop=True)
    C = ML_competitors_all.copy()
    ML_competitors_all = ML_competitors_all.drop("model", axis=1).groupby(by=["db", "imputer", "fusion_strategy", "train_percentage", "test_percentage"]).mean().round(2).reset_index()
    ML_competitors_all.loc[:, "model"] = "ML w imputer"
    ML_competitors_all.loc[ML_competitors_all.imputer == "noimputation", "model"] = "ML w/o imputer"
    NAIM_vs_ML_all = pd.concat([NAIM_results_all, ML_competitors_all], axis=0).reset_index(drop=True)
    del ML_competitors_all

    ML_competitors_modalities = ML_competitors_results.loc[ML_competitors_results.missing_strategy == "modalities"].drop("missing_strategy", axis=1)
    D = ML_competitors_modalities.copy()
    ML_competitors_modalities = ML_competitors_modalities.drop("model", axis=1).groupby(by=["db", "imputer", "fusion_strategy", "train_percentage", "test_percentage"]).mean().round(2).reset_index()
    ML_competitors_modalities.loc[:, "model"] = "ML w imputer"
    ML_competitors_modalities.loc[ML_competitors_modalities.imputer == "noimputation", "model"] = "ML w/o imputer"
    NAIM_vs_ML_modalities = pd.concat([NAIM_results_modalities, ML_competitors_modalities], axis=0).reset_index(drop=True)
    del ML_competitors_modalities, ML_competitors_results

    ####################################################################################################################

    # NAIM vs NAIM
    NAIM_competitors_all = NAIM_competitors_results.loc[NAIM_competitors_results.missing_strategy == "all"].drop("missing_strategy", axis=1)
    NAIM_competitors_all = NAIM_competitors_all.loc[NAIM_competitors_all.apply(lambda x: train_percentage_rule(x, all_original_missing_percentages), axis=1)].reset_index(drop=True)
    NAIM_competitors_all = NAIM_competitors_all.loc[NAIM_competitors_all.apply(lambda x: test_percentage_rule(x, all_original_missing_percentages), axis=1)].reset_index(drop=True)
    E = NAIM_competitors_all.copy()
    NAIM_competitors_all = NAIM_competitors_all.drop("model", axis=1).groupby(by=["db", "imputer", "fusion_strategy", "train_percentage", "test_percentage"]).mean().round(2).reset_index()
    NAIM_competitors_all.loc[:, "model"] = "NAIM"
    NAIM_vs_NAIM_all = pd.concat([NAIM_results_all, NAIM_competitors_all], axis=0).reset_index(drop=True)
    del NAIM_competitors_all

    NAIM_competitors_modalities = NAIM_competitors_results.loc[NAIM_competitors_results.missing_strategy == "modalities"].drop("missing_strategy", axis=1)

    NAIM_competitors_modalities = NAIM_competitors_modalities.drop("model", axis=1).groupby(by=["db", "imputer", "fusion_strategy", "train_percentage", "test_percentage"]).mean().round(2).reset_index()
    NAIM_competitors_modalities.loc[:, "model"] = "NAIM"
    NAIM_vs_NAIM_modalities = pd.concat([NAIM_results_modalities, NAIM_competitors_modalities], axis=0).reset_index(drop=True)
    del NAIM_competitors_modalities, NAIM_competitors_results

    AC_mean = pd.concat([A, C, NAIM_results_all], axis=0, ignore_index=True)
    AC_auc = AC_mean.loc[AC_mean.db.isin([k for k, v in dbs_metrics.items() if v == "auc"])].drop("db", axis=1).groupby(by=[ "fusion_strategy", "model", "imputer", "train_percentage", "test_percentage"]).agg("mean").round(2)["auc"].unstack(-2).unstack()
    AC_mcc = AC_mean.loc[AC_mean.db.isin([k for k, v in dbs_metrics.items() if v == "mcc"])].drop("db", axis=1).groupby(by=[ "fusion_strategy", "model", "imputer", "train_percentage", "test_percentage"]).agg("mean").round(2)["mcc"].unstack(-2).unstack()

    BD_mean = pd.concat([B, D, NAIM_results_modalities], axis=0, ignore_index=True)
    BD_auc = BD_mean.loc[BD_mean.db.isin([k for k, v in dbs_metrics.items() if v == "auc"])].drop("db", axis=1).groupby(by=[ "fusion_strategy", "model", "imputer", "train_percentage", "test_percentage"]).agg("mean").round(2)["auc"].unstack(-2).unstack()
    BD_mcc = BD_mean.loc[BD_mean.db.isin([k for k, v in dbs_metrics.items() if v == "mcc"])].drop("db", axis=1).groupby(by=[ "fusion_strategy", "model", "imputer", "train_percentage", "test_percentage"]).agg("mean").round(2)["mcc"].unstack(-2).unstack()

    rf, naim = 0, 0
    for a in [AC_auc, AC_mcc, BD_auc, BD_mcc]:
        for col in a.columns:
            x = a[col].idxmax()
            if x[1] == "RandomForest":
                rf += 1
            if x[1] == "NAIM":
                naim += 1
            print( a[col].idxmax())
        print("\n\n\n")
    AC_auc.round(2).to_latex(os.path.join(plots_output_path, "ALL_auc.txt"))
    AC_auc.to_csv(os.path.join(plots_output_path, "ALL_auc.csv"))
    AC_mcc.round(2).to_latex(os.path.join(plots_output_path, "ALL_mcc.txt"))
    AC_mcc.to_csv(os.path.join(plots_output_path, "ALL_mcc.csv"))
    BD_auc.round(2).to_latex(os.path.join(plots_output_path, "MOD_auc.txt"))
    BD_auc.to_csv(os.path.join(plots_output_path, "MOD_auc.csv"))
    BD_mcc.round(2).to_latex(os.path.join(plots_output_path, "MOD_mcc.txt"))
    BD_mcc.to_csv(os.path.join(plots_output_path, "MOD_mcc.csv"))
    plot_performance(NAIM_vs_DL_all, metrics=dbs_metrics, filename=f"NAIM_vs_DL_all", output_path=plots_output_path, extension=extension)
    plot_performance(NAIM_vs_DL_modalities, metrics=dbs_metrics, filename=f"NAIM_vs_DL_modalities", output_path=plots_output_path, extension=extension)

    plot_performance(NAIM_vs_ML_all, metrics=dbs_metrics, filename=f"NAIM_vs_ML_all", output_path=plots_output_path, extension=extension)
    plot_performance(NAIM_vs_ML_modalities, metrics=dbs_metrics, filename=f"NAIM_vs_ML_modalities", output_path=plots_output_path, extension=extension)

    plot_performance(NAIM_vs_NAIM_all, metrics=dbs_metrics, filename=f"NAIM_vs_NAIM_all", output_path=plots_output_path, extension=extension)
    plot_performance(NAIM_vs_NAIM_modalities, metrics=dbs_metrics, filename=f"NAIM_vs_NAIM_modalities", output_path=plots_output_path, extension=extension)


if __name__ == "__main__":
    base_path = "/Volumes/Aaron SSD/UCBM/Projects"
    results_paths = ["multiNAIM_ADNI/ADNI_diagnosis_binary_42_multimodal_early_fusion_classification_with_missing_generation",
                    "multiNAIM_ADNI/ADNI_diagnosis_binary_42_multimodal_joint_fusion_classification_with_missing_generation",
                    "multiNAIM_ADNI/ADNI_diagnosis_binary_42_multimodal_late_fusion_classification_with_missing_generation",
                    "multiNAIM_ADNI/ADNI_diagnosis_multiclass_42_multimodal_early_fusion_classification_with_missing_generation",
                    "multiNAIM_ADNI/ADNI_diagnosis_multiclass_42_multimodal_joint_fusion_classification_with_missing_generation",
                    "multiNAIM_ADNI/ADNI_diagnosis_multiclass_42_multimodal_late_fusion_classification_with_missing_generation",
                    "multiNAIM_AI4Covid/AI4Covid_death_42_multimodal_early_fusion_classification_with_missing_generation",
                    "multiNAIM_AI4Covid/AI4Covid_death_42_multimodal_joint_fusion_classification_with_missing_generation",
                    "multiNAIM_AI4Covid/AI4Covid_death_42_multimodal_late_fusion_classification_with_missing_generation",
                    "multiNAIM_AI4Covid/AI4Covid_prognosis_42_multimodal_early_fusion_classification_with_missing_generation",
                    "multiNAIM_AI4Covid/AI4Covid_prognosis_42_multimodal_joint_fusion_classification_with_missing_generation",
                    "multiNAIM_AI4Covid/AI4Covid_prognosis_42_multimodal_late_fusion_classification_with_missing_generation",
                    "multiNAIM_ADNI/ADNI_prognosis_m12_42_multimodal_early_fusion_classification_with_missing_generation",
                    "multiNAIM_ADNI/ADNI_prognosis_m12_42_multimodal_joint_fusion_classification_with_missing_generation",
                    "multiNAIM_ADNI/ADNI_prognosis_m12_42_multimodal_late_fusion_classification_with_missing_generation",
                    "multiNAIM_ADNI/ADNI_prognosis_m24_42_multimodal_early_fusion_classification_with_missing_generation",
                    "multiNAIM_ADNI/ADNI_prognosis_m24_42_multimodal_joint_fusion_classification_with_missing_generation",
                    "multiNAIM_ADNI/ADNI_prognosis_m24_42_multimodal_late_fusion_classification_with_missing_generation",
                    "multiNAIM_ADNI/ADNI_prognosis_m36_42_multimodal_early_fusion_classification_with_missing_generation",
                    "multiNAIM_ADNI/ADNI_prognosis_m36_42_multimodal_joint_fusion_classification_with_missing_generation",
                    "multiNAIM_ADNI/ADNI_prognosis_m36_42_multimodal_late_fusion_classification_with_missing_generation",
                    "multiNAIM_ADNI/ADNI_prognosis_m48_42_multimodal_early_fusion_classification_with_missing_generation",
                    "multiNAIM_ADNI/ADNI_prognosis_m48_42_multimodal_joint_fusion_classification_with_missing_generation",
                    "multiNAIM_ADNI/ADNI_prognosis_m48_42_multimodal_late_fusion_classification_with_missing_generation"]

    results_paths = [os.path.join(base_path, path) for path in results_paths]

    plots_output_path = "/Users/camillocaruso/Downloads/plots"

    imputers = ["knn"]

    plot_missing_results(results_paths, plots_output_path, imputers, extension="pdf")
