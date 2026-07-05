import logging
from typing import Union, List
from hydra.utils import call
from omegaconf import DictConfig, OmegaConf
from CMC_utils.miscellaneous import join_preprocessing_params, recursive_cfg_substitute
from CMC_utils.datasets import MultimodalJointFusionTabularDatasetTorch

__all__ = ["set_multimodal_params"]


log = logging.getLogger(__name__)

def set_multimodal_params(model_cfg: dict, preprocessing_params: List[Union[dict, DictConfig]], train_set: MultimodalJointFusionTabularDatasetTorch, **_) -> dict:
    """
    Set the parameters of the Multimodal model
    Parameters
    ----------
    model_cfg : dict
    preprocessing_params : List[Union[dict, DictConfig]], optional
    train_set : SupervisedTabularDatasetTorch

    Returns
    -------
    dict
        model_cfg
    """
    for ms_model, params, fset in zip(model_cfg["init_params"]["ms_models"].keys(), preprocessing_params, train_set.datasets):
        model_cfg["init_params"]["ms_models"][ms_model] = call(model_cfg["init_params"]["ms_models"][ms_model]["set_params_function"], OmegaConf.create(model_cfg["init_params"]["ms_models"][ms_model]), preprocessing_params=params, train_set=fset)

    model_cfg["init_params"]["shared_net"] = call(model_cfg["init_params"]["shared_net"]["set_params_function"], OmegaConf.create(model_cfg["init_params"]["shared_net"]), preprocessing_params=join_preprocessing_params(*preprocessing_params), train_set=train_set)

    return model_cfg


if __name__ == "__main__":
    pass
