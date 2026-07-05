import os
import logging
import numpy as np
import pandas as pd
from hydra.utils import call
from CMC_utils.miscellaneous import do_nothing
from CMC_utils.preprocessing import discrete_to_label
from CMC_utils.save_load import save_table
from CMC_utils.datasets import SupervisedTabularDatasetTorch

log = logging.getLogger(__name__)

__all__ = ["test_sklearn_model"]


def infer_classification_model(model) -> callable:
    """
    Infer the predict function of a sklearn model
    Parameters
    ----------
    model : sklearn model

    Returns
    -------
    function
    """
    return model.predict_proba


def test_sklearn_model(*sets: SupervisedTabularDatasetTorch, model_params: dict, model_path: str, prediction_path: str, test_fold: int = 0, val_fold: int = 0, **kwargs) -> None:
    """
    Test a sklearn model
    Parameters
    ----------
    sets : SupervisedTabularDatasetTorch
    model_params : dict
    model_path : str
    prediction_path : str
    test_fold : int
    val_fold : int
    kwargs : dict

    Returns
    -------
    None
    """
    model_path = os.path.join( model_path, f"{test_fold}_{val_fold}.{model_params['file_extension']}")
    model = call(model_params["load_function"], model_path)

    model_options = dict(binary=infer_classification_model, discrete=infer_classification_model)
    model_params_options = dict(binary={}, discrete={})

    output_options = dict(binary=do_nothing, discrete=do_nothing)
    output_params_options = dict(binary={}, discrete={})

    labels_options = dict(binary=discrete_to_label, discrete=discrete_to_label)

    labels_params_options = dict(binary={"classes": sets[0].classes}, discrete={"classes": sets[0].classes})

    for fset in sets:

        filename = f"{test_fold}_{val_fold}_{fset.set_name}.csv"
        if os.path.exists(os.path.join(prediction_path, filename)):
            continue

        data, labels, ID = fset.get_data()

        # probs = model_options[sets[0].label_type](model)(data.astype(float), **model_params_options[sets[0].label_type])
        data = data.astype("float32")
        probs = model_options[fset.label_type](model)(data, **model_params_options[fset.label_type])

        # probs = output_options[sets[0].label_type](probs, model=model, **output_params_options[sets[0].label_type])
        probs = output_options[fset.label_type](probs, model=model, **output_params_options[fset.label_type])

        # preds = list(map( lambda label: labels_options[sets[0].label_type](label, **labels_params_options[sets[0].label_type]), probs))
        preds = list(map( lambda label: labels_options[fset.label_type](label, **labels_params_options[fset.label_type]), probs))

        if fset.label_type == "binary":
            probs = probs[:, 1]

        labels = list(map( lambda label: labels_options[fset.label_type](label, **labels_params_options[fset.label_type]), labels))

        results = pd.DataFrame( dict( ID=ID, label=labels, prediction=preds, probability=probs.tolist() ))
        save_table( results, filename, prediction_path )
        log.info("Inference done")


if __name__ == "__main__":
    pass
