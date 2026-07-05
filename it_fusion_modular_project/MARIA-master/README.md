# MARIA
[![DOI](https://img.shields.io/badge/DOI-10.1016/j.compbiomed.2025.110843-blue.svg)](https://doi.org/10.1016/j.compbiomed.2025.110843)

1. [Installation](#installation)
2. [Usage](#usage)
   1. [Reproducing the experiments](#repr_exp)
   2. [Train & Test on your dataset](#new_exp)
      1. [Experiment declaration](#exp_decl)
      2. [Dataset preparation](#data_prep)
      3. [Experiment configuration](#exp_conf)
3. [Contact](#contact)
4. [Citation](#citation)

---

This document describes the implementation of *``MARIA: A multimodal transformer model for incomplete healthcare data´´* ([MARIA](https://doi.org/10.1016/j.compbiomed.2025.110843)) in PyTorch.
MARIA is a multimodal learning framework specifically designed for the analysis of biomedical tabular data, with a focus on addressing missing values across multiple data modalities without the need for any imputation strategy.

Building upon [NAIM](https://github.com/cosbidev/NAIM), MARIA extends the attention-based missing value handling to the multimodal setting, where each modality is processed by a dedicated modality-specific network.
The framework supports three fusion strategies to combine information from different modalities:

- **Early Fusion**: features from all modalities are concatenated before being fed into a single model.
- **Joint Fusion**: each modality is processed by a dedicated network, and their intermediate representations are combined through a shared network.
- **Late Fusion**: each modality is processed independently and the final predictions are combined at decision level.

At every epoch, features are randomly masked (MCAR - Missing Completely At Random) across modalities to prevent co-adaptations among features and to enhance the model's generalization capability, enabling robust performance even in the presence of high percentages of missing values.

---

# Installation <div id='installation'/>
We used Python 3.9 for the development of the code.
To install the required packages, it is sufficient to run the following command:
```bash
pip install -r requirements.txt
```
and install a version of PyTorch compatible with the device available. We used torch==1.13.0.

---

# Usage <div id='usage'/>
The execution of the code heavily relies on Facebook's [Hydra](https://hydra.cc/) library.
Specifically, through a multitude of configuration files that define every aspect of the experiment, it is possible to
conduct the desired experiment without modifying the code.
These configuration files have a hierarchical structure through which they are composed into a single configuration
file that serves as input to the program.
More specifically, the [`main.py`](./main.py) file will call the [`config.yaml`](./confs/config.yaml) file, from which
the configuration files tree begins.

The framework supports five pipeline types:

| Pipeline | Description |
|----------|-------------|
| `simple` | Single modality classification using all available data |
| `missing` | Single modality classification with missing data generation |
| `multimodal_early_fusion` | Early fusion multimodal classification with missing data generation |
| `multimodal_joint_fusion` | Joint fusion multimodal classification with missing data generation |
| `multimodal_late_fusion` | Late fusion multimodal classification with missing data generation |

## Reproducing the experiments <div id='repr_exp'/>

The experiments presented in the paper use two biomedical datasets, each comprising multiple modalities:

<div style="display: block; margin: 0 auto; text-align: center">

| Dataset | Task | Modalities |
|---------|------|------------|
| ADNI | Diagnosis (CN, EMCI, LMCI, AD) | Assessment, Biospecimen, Image Analysis, Subject Characteristics |
| ADNI | Prognosis | Assessment, Biospecimen, Image Analysis, Subject Characteristics |
| AI4Covid | Death prediction | Blood, History, Personal, State |
| AI4Covid | Prognosis | Blood, History, Personal, State |

</div>

The experiment configuration files for each dataset, task, and fusion strategy are available in the [`./confs/experiment/`](./confs/experiment/) folder.
For example, to run the ADNI diagnosis experiment with joint fusion using NAIM:

```bash
python main.py experiment=ADNI_diagnosis_joint_NAIM
```

Similarly, for AI4Covid death prediction with early fusion:

```bash
python main.py experiment=AI4Covid_early_death
```

The available experiment configurations follow the naming convention `<Dataset>_<task>_<fusion>_<model>.yaml` for joint fusion experiments and `<Dataset>_<fusion>_<task>.yaml` for early and late fusion experiments:

<div style="display: block; margin: 0 auto; text-align: center">

| Dataset | Task | Fusion | Config file |
|---------|------|--------|-------------|
| ADNI | Diagnosis | Early | [`ADNI_diagnosis_early.yaml`](./confs/experiment/ADNI_diagnosis_early.yaml) |
| ADNI | Diagnosis | Joint (NAIM) | [`ADNI_diagnosis_joint_NAIM.yaml`](./confs/experiment/ADNI_diagnosis_joint_NAIM.yaml) |
| ADNI | Diagnosis | Joint (FTTransformer) | [`ADNI_diagnosis_joint_FTTransformer.yaml`](./confs/experiment/ADNI_diagnosis_joint_FTTransformer.yaml) |
| ADNI | Diagnosis | Joint (MLP) | [`ADNI_diagnosis_joint_MLP.yaml`](./confs/experiment/ADNI_diagnosis_joint_MLP.yaml) |
| ADNI | Diagnosis | Joint (TabNet) | [`ADNI_diagnosis_joint_TabNet.yaml`](./confs/experiment/ADNI_diagnosis_joint_TabNet.yaml) |
| ADNI | Diagnosis | Joint (TabTransformer) | [`ADNI_diagnosis_joint_TabTransformer.yaml`](./confs/experiment/ADNI_diagnosis_joint_TabTransformer.yaml) |
| ADNI | Diagnosis | Late | [`ADNI_diagnosis_late.yaml`](./confs/experiment/ADNI_diagnosis_late.yaml) |
| ADNI | Prognosis | Early | [`ADNI_prognosis_early.yaml`](./confs/experiment/ADNI_prognosis_early.yaml) |
| ADNI | Prognosis | Joint (NAIM) | [`ADNI_prognosis_joint_NAIM.yaml`](./confs/experiment/ADNI_prognosis_joint_NAIM.yaml) |
| ADNI | Prognosis | Joint (FTTransformer) | [`ADNI_prognosis_joint_FTTransformer.yaml`](./confs/experiment/ADNI_prognosis_joint_FTTransformer.yaml) |
| ADNI | Prognosis | Joint (MLP) | [`ADNI_prognosis_joint_MLP.yaml`](./confs/experiment/ADNI_prognosis_joint_MLP.yaml) |
| ADNI | Prognosis | Joint (TabNet) | [`ADNI_prognosis_joint_TabNet.yaml`](./confs/experiment/ADNI_prognosis_joint_TabNet.yaml) |
| ADNI | Prognosis | Joint (TabTransformer) | [`ADNI_prognosis_joint_TabTransformer.yaml`](./confs/experiment/ADNI_prognosis_joint_TabTransformer.yaml) |
| ADNI | Prognosis | Late | [`ADNI_prognosis_late.yaml`](./confs/experiment/ADNI_prognosis_late.yaml) |
| AI4Covid | Death | Early | [`AI4Covid_early_death.yaml`](./confs/experiment/AI4Covid_early_death.yaml) |
| AI4Covid | Death | Joint (NAIM) | [`AI4Covid_joint_death_NAIM.yaml`](./confs/experiment/AI4Covid_joint_death_NAIM.yaml) |
| AI4Covid | Death | Joint (FTTransformer) | [`AI4Covid_joint_death_FTTransformer.yaml`](./confs/experiment/AI4Covid_joint_death_FTTransformer.yaml) |
| AI4Covid | Death | Joint (MLP) | [`AI4Covid_joint_death_MLP.yaml`](./confs/experiment/AI4Covid_joint_death_MLP.yaml) |
| AI4Covid | Death | Joint (TabNet) | [`AI4Covid_joint_death_TabNet.yaml`](./confs/experiment/AI4Covid_joint_death_TabNet.yaml) |
| AI4Covid | Death | Joint (TabTransformer) | [`AI4Covid_joint_death_TabTransformer.yaml`](./confs/experiment/AI4Covid_joint_death_TabTransformer.yaml) |
| AI4Covid | Death | Late | [`AI4Covid_late_death.yaml`](./confs/experiment/AI4Covid_late_death.yaml) |
| AI4Covid | Prognosis | Early | [`AI4Covid_early_prognosis.yaml`](./confs/experiment/AI4Covid_early_prognosis.yaml) |
| AI4Covid | Prognosis | Joint (NAIM) | [`AI4Covid_joint_prognosis_NAIM.yaml`](./confs/experiment/AI4Covid_joint_prognosis_NAIM.yaml) |
| AI4Covid | Prognosis | Joint (FTTransformer) | [`AI4Covid_joint_prognosis_FTTransformer.yaml`](./confs/experiment/AI4Covid_joint_prognosis_FTTransformer.yaml) |
| AI4Covid | Prognosis | Joint (MLP) | [`AI4Covid_joint_prognosis_MLP.yaml`](./confs/experiment/AI4Covid_joint_prognosis_MLP.yaml) |
| AI4Covid | Prognosis | Joint (TabNet) | [`AI4Covid_joint_prognosis_TabNet.yaml`](./confs/experiment/AI4Covid_joint_prognosis_TabNet.yaml) |
| AI4Covid | Prognosis | Joint (TabTransformer) | [`AI4Covid_joint_prognosis_TabTransformer.yaml`](./confs/experiment/AI4Covid_joint_prognosis_TabTransformer.yaml) |
| AI4Covid | Prognosis | Late | [`AI4Covid_late_prognosis.yaml`](./confs/experiment/AI4Covid_late_prognosis.yaml) |

</div>

These experiments generate different percentages of missing values (MCAR) in the training and testing sets.
Specifically, the percentages used are indicated by `missing_percentages` in each experiment configuration file (e.g., `[0.0, 0.05, 0.1, 0.3, 0.5, 0.75]`).

For single-modality experiments with missing data generation, the [`classification_with_missing_generation.yaml`](./confs/experiment/classification_with_missing_generation.yaml) configuration can be used:

```bash
python main.py experiment=classification_with_missing_generation experiment/databases@db=ADNI_diagnosis_assessment
```

For each experiment, this code produces a folder named `<experiment-name>/<experiment-subname>` which contains everything generated by the code.
In particular, the following folders and files are present:

1. `cross_validation`: this folder contains a folder for each training fold, indicated as a composition of test and validation folds `<test-fold>_<val-fold>`, reporting the information on the train, validation and test sets in 3 separate csv files.
2. `preprocessing`: this folder contains all the preprocessing information divided into 3 main folders:
   1. `numerical_preprocessing`: in this folder, for each percentage of missing values considered, there is a csv file for each fold reporting the information on the preprocessing params of numerical features.
   2. `categorical_preprocessing`: in this folder, for each percentage of missing values considered, there is a csv file for each fold reporting the information on the preprocessing params of categorical features.
   3. `imputer`: in this folder, for each percentage of missing values considered, there are csv files for each fold with information on the imputation strategy applied to handle missing values and a pkl file containing the imputer fitted on the training data of the fold.
3. `saved_models`: this folder contains, for each percentage of missing values considered, a folder with the model's name that includes, for each fold, a csv file with the model's parameters and a pkl or pth file containing the trained model.
4. `predictions`: this folder contains, for each percentage of missing values considered, a folder that reports the predictions obtained from the training and validation sets and separately those of the test set.
5. `results`: this folder reports, for each percentage of missing values considered, the performance on the train, validation, and test sets separately. Specifically, for each set two folders named `balanced` and `unbalanced` containing the performance are reported, presented in 3 separate files with increasing levels of averaging:
   1. `all_test_performance.csv`: performance evaluated for each fold and each class.
   2. `classes_average_performance.csv`: average performance of the folds for each class.
   3. `set_average_performance.csv`: average performance across folds and classes.
6. `config.yaml`: this file contains the configuration file used as input for the experiment.
7. `<experiment-name>.log`: this is the log file of the experiment.

> NOTE: In case an experiment should be interrupted, voluntarily or not, it is possible to resume it from where it was interrupted by setting the `continue_experiment` parameter to `True` in the experiment configuration file.

## Train & Test on your dataset <div id='new_exp'/>

### Experiment declaration  <div id='exp_decl'/>

As mentioned above, the experiment configuration file is created at the time of code execution starting from the
[`config.yaml`](./confs/config.yaml) file, in which the configuration file for the experiment to be performed is declared,
along with the device to use and the system paths configuration.

```yaml
device: cuda # cpu, cuda, or mps

defaults:
  - _self_
  - experiment: multimodal_joint_fusion_classification_with_missing_generation # Experiment to perform
  - experiment/paths/system@: local # System paths configuration
```

The possible options for the `experiment` parameter are the experiment configuration files contained in the [`./confs/experiment/`](./confs/experiment/) folder.
The main experiment types are:
- [`classification`](./confs/experiment/classification.yaml): single modality classification
- [`classification_with_missing_generation`](./confs/experiment/classification_with_missing_generation.yaml): single modality classification with MCAR missing data generation
- [`multimodal_early_fusion_classification_with_missing_generation`](./confs/experiment/multimodal_early_fusion_classification_with_missing_generation.yaml): early fusion multimodal classification
- [`multimodal_joint_fusion_classification_with_missing_generation`](./confs/experiment/multimodal_joint_fusion_classification_with_missing_generation.yaml): joint fusion multimodal classification
- [`multimodal_late_fusion_classification_with_missing_generation`](./confs/experiment/multimodal_late_fusion_classification_with_missing_generation.yaml): late fusion multimodal classification

### Dataset preparation <div id='data_prep'/>

To prepare a dataset for the analysis with this code, it is sufficient to prepare a configuration file, specific for the
dataset, similar to those already provided in the folder [`./confs/experiment/databases`](./confs/experiment/databases).
The path to the data must be specified in the `path` parameter in the dataset's configuration file.
Thanks to the [interpolation](https://hydra.cc/docs/patterns/specializing_config/) functionality of Hydra the path can be composed using the `${data_path}` interpolation key.
Once the dataset configuration file is prepared, it is important that it is placed in the [`./confs/experiment/databases`](./confs/experiment/databases) folder.
In particular, it is important that the dataset configuration file is structured as follows:

```yaml
_target_: CMC_utils.datasets.ClassificationDataset # DO NOT CHANGE
_convert_: all # DO NOT CHANGE

name: <dataset-name> # Name of the dataset
db_type: tabular # DO NOT CHANGE

classes: ["<class-1-name>", ..., "<class-n-name>"] # List of the classes
label_type: multiclass # multiclass or binary

task: classification # DO NOT CHANGE

path: ${data_path}/<relative-path-to-file> # Relative path to the file

columns: # Dictionary containing features names as keys and their types as values
  <ID-name>:        id             # Name of the ID column if present
  <feature-1-name>: <feature-type> # int, float or category
  <feature-2-name>: <feature-type> # int, float or category
  # Other features to be inserted
  <label-name>:     target         # Name of the target column

pandas_load_kwargs:
  na_values: [ "?" ]
  header: 0
  index_col: 0

dataset_class: # DO NOT CHANGE
  _target_: CMC_utils.datasets.SupervisedTabularDatasetTorch # DO NOT CHANGE
  _convert_: all # DO NOT CHANGE
```

In the `columns` definition, `id` and `target` feature types can be used to define the ID and classes columns respectively.

For multimodal experiments, multiple dataset configuration files must be prepared (one per modality) and referenced in the experiment configuration file using the `databases@dbs.<index>` key.

### Experiment configuration <div id='exp_conf'/>

The experiment configuration file defines the specifics for conducting the desired pipeline.
It begins with some general information, such as the name of the experiment, the pipeline to be executed, the seed for
randomness control, training verbosity, and the percentages of missing values to be tested.

```yaml
experiment_name: ${db_name}_${seed}_multimodal_joint_fusion_classification_with_missing_generation
pipeline: multimodal_joint_fusion

seed: 42
verbose: 1
continue_experiment: False

missing_percentages: [0.0, 0.05, 0.1, 0.25, 0.5]
```

Then, all other necessary configuration files for the different parts of the experiment are declared.
The possible options for each part are listed in the table below.

| Params                    | Keys                                                                                | Options |
|---------------------------|-------------------------------------------------------------------------------------|---------|
| Dataset                   | `databases@db` or `databases@dbs.<index>`                                           | [ADNI_diagnosis_assessment](./confs/experiment/databases/ADNI_diagnosis_assessment.yaml), [ADNI_diagnosis_biospecimen](./confs/experiment/databases/ADNI_diagnosis_biospecimen.yaml), [ADNI_diagnosis_imageanalysis](./confs/experiment/databases/ADNI_diagnosis_imageanalysis.yaml), [ADNI_diagnosis_subjectcharacteristics](./confs/experiment/databases/ADNI_diagnosis_subjectcharacteristics.yaml), [ADNI_prognosis_assessment](./confs/experiment/databases/ADNI_prognosis_assessment.yaml), [ADNI_prognosis_biospecimen](./confs/experiment/databases/ADNI_prognosis_biospecimen.yaml), [ADNI_prognosis_imageanalysis](./confs/experiment/databases/ADNI_prognosis_imageanalysis.yaml), [ADNI_prognosis_subjectcharacteristics](./confs/experiment/databases/ADNI_prognosis_subjectcharacteristics.yaml), [AI4Covid_blood_death](./confs/experiment/databases/AI4Covid_blood_death.yaml), [AI4Covid_blood_prognosis](./confs/experiment/databases/AI4Covid_blood_prognosis.yaml), [AI4Covid_history_death](./confs/experiment/databases/AI4Covid_history_death.yaml), [AI4Covid_history_prognosis](./confs/experiment/databases/AI4Covid_history_prognosis.yaml), [AI4Covid_personal_death](./confs/experiment/databases/AI4Covid_personal_death.yaml), [AI4Covid_personal_prognosis](./confs/experiment/databases/AI4Covid_personal_prognosis.yaml), [AI4Covid_state_death](./confs/experiment/databases/AI4Covid_state_death.yaml), [AI4Covid_state_prognosis](./confs/experiment/databases/AI4Covid_state_prognosis.yaml) |
| Cross Validation          | `cross_validation@test_cv` `cross_validation@val_cv`                                | [bootstrap](./confs/experiment/cross_validation/bootstrap.yaml), [holdout](./confs/experiment/cross_validation/holdout.yaml), [kfold](./confs/experiment/cross_validation/kfold.yaml), [leave_one_out](./confs/experiment/cross_validation/leave_one_out.yaml), [predefined](./confs/experiment/cross_validation/predefined.yaml), [stratifiedkfold](./confs/experiment/cross_validation/stratifiedkfold.yaml) |
| Numerical Preprocessing   | `preprocessing/numerical`                                                           | [normalize](./confs/experiment/preprocessing/numerical/normalize.yaml), [standardize](./confs/experiment/preprocessing/numerical/standardize.yaml) |
| Categorical Preprocessing | `preprocessing/categorical`                                                         | [categoricalencode](./confs/experiment/preprocessing/categorical/categoricalencode.yaml), [onehotencode](./confs/experiment/preprocessing/categorical/onehotencode.yaml) |
| Imputation Strategy       | `preprocessing/imputer`                                                             | [simple](./confs/experiment/preprocessing/imputer/simple.yaml), [knn](./confs/experiment/preprocessing/imputer/knn.yaml), [iterative](./confs/experiment/preprocessing/imputer/iterative.yaml), [noimputation](./confs/experiment/preprocessing/imputer/noimputation.yaml) |
| Missing Generation        | `preprocessing/missing_generation`                                                  | [MCAR_global](./confs/experiment/preprocessing/missing_generation/MCAR_global.yaml), [MCAR_feature](./confs/experiment/preprocessing/missing_generation/MCAR_feature.yaml), [MCAR_sample](./confs/experiment/preprocessing/missing_generation/MCAR_sample.yaml), [no_generation](./confs/experiment/preprocessing/missing_generation/no_generation.yaml) |
| Model                     | `model`                                                                             | [naim](./confs/experiment/model/naim.yaml), [multimodal_learner](./confs/experiment/model/multimodal_learner.yaml), [adaboost](./confs/experiment/model/adaboost.yaml), [custom_mlp](./confs/experiment/model/custom_mlp.yaml), [dt](./confs/experiment/model/dt.yaml), [fttransformer](./confs/experiment/model/fttransformer.yaml), [histgradientboostingtree](./confs/experiment/model/histgradientboostingtree.yaml), [mlp_sklearn](./confs/experiment/model/mlp_sklearn.yaml), [rf](./confs/experiment/model/rf.yaml), [svm](./confs/experiment/model/svm.yaml), [tabnet](./confs/experiment/model/tabnet.yaml), [tabtransformer](./confs/experiment/model/tabtransformer.yaml), [xgboost](./confs/experiment/model/xgboost.yaml) |
| Metrics                   | `metric@train.set_metrics.<metric-name>` `metric@performance_metrics.<metric-name>` | [auc](./confs/experiment/metric/auc.yaml), [accuracy](./confs/experiment/metric/accuracy.yaml), [recall](./confs/experiment/metric/recall.yaml), [precision](./confs/experiment/metric/precision.yaml), [f1_score](./confs/experiment/metric/f1_score.yaml), [mcc](./confs/experiment/metric/mcc.yaml), [gmean](./confs/experiment/metric/gmean.yaml) |

To modify some of the hyperparameters of the models, it is possible to modify the [`ml_params`](./confs/experiment/model_type_params/ml_params.yaml) and [`dl_params`](./confs/experiment/model_type_params/dl_params.yaml) files.
For the ML models it is possible to define the number of estimators (`n_estimators`), whereas for the DL models it is possible to define the number of epochs (`max_epochs`), the warm-up number of epochs (`min_epochs`),
the batch size (`batch_size`), the early stopping's (`early_stopping_patience`) and the scheduler's (`scheduler_patience`) patience and their tolerance for performance improvement (`performance_tolerance`), the device to use for training (`device`).
It is also possible to define the learning rates to be tested (`learning_rates`), but to be compatible with some of the competitors available in the models list, it is necessary to define also the initial learning rate (`init_learning_rate`) and the final learning rate (`end_learning_rate`).

#### [./confs/experiment/model_type_params/ml_params.yaml](./confs/experiment/model_type_params/ml_params.yaml)
```yaml
n_estimators: 100 # Number of estimators for the ML models
```

#### [./confs/experiment/model_type_params/dl_params.yaml](./confs/experiment/model_type_params/dl_params.yaml)
```yaml
max_epochs: 1500 # Maximum number of epochs
min_epochs: 50 # Warm-up number of epochs
batch_size: 32 # Batch size
init_learning_rate: 1e-3 # Initial learning rate
end_learning_rate: 1e-8 # Final learning rate
learning_rates: [1e-3, 1e-4, 1e-5, 1e-6, 1e-7] # Learning rates for the scheduler
early_stopping_patience: 50 # Patience for the early stopping
scheduler_patience: 25 # Patience for the scheduler
performance_tolerance: 1e-3 # Tolerance for the performance improvement
device: cuda # cpu or cuda or mps, device to use for training
```

---
# Contact <div id='contact'/>

For any questions, please contact [camillomaria.caruso@unicampus.it](mailto:camillomaria.caruso@unicampus.it) and [valerio.guarrasi@unicampus.it](mailto:valerio.guarrasi@unicampus.it).

---

# Citation <div id='citation'/>

```bibtex
@article{caruso2025maria,
  title={MARIA: A multimodal transformer model for incomplete healthcare data},
  author={Caruso, Camillo Maria and Soda, Paolo and Guarrasi, Valerio},
  journal={Computers in Biology and Medicine},
  volume={196},
  pages={110843},
  year={2025},
  publisher={Elsevier},
  doi={10.1016/j.compbiomed.2025.110843},
} 

```
