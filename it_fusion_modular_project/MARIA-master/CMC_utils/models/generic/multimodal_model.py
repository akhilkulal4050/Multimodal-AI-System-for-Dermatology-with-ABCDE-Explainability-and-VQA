import torch
from hydra.utils import instantiate
from CMC_utils.miscellaneous import recursive_cfg_substitute


__all__ = ["MultimodalLearner"]


class MultimodalLearner(torch.nn.Module):
    """
    MultimodalLearner is a generic class for multimodal learning. It takes as input a list of models and outputs a
    single prediction. The models can be of any type, as long as they have a forward method that takes a single input
    and returns a single output. The output of the models is concatenated and fed to a fully connected layer.
    """
    def __init__(self, ms_models, shared_net, **_):
        super(MultimodalLearner, self).__init__()

        for model_id in ms_models.keys():
            if ms_models[model_id]["name"].startswith("TabNet"):
                param_dict = dict(output_dim=ms_models[model_id]["init_params"].get("input_dim", None))
            elif ms_models[model_id]["name"].startswith(("TabTransformer", "FTTransformer")):
                param_dict = dict( dim_out=ms_models[model_id]["init_params"].get("num_continuous", 0) + len(ms_models[model_id]["init_params"].get("cat_idxs", [])), extractor=True )
            elif ms_models[model_id]["name"].startswith("naim"):
                param_dict = dict(extractor=True)
            elif ms_models[model_id]["name"].startswith("MLP"):
                param_dict = dict(extractor=True)
            else:
                param_dict = dict()

            ms_models[model_id] = recursive_cfg_substitute(ms_models[model_id], param_dict)

        self.ms_models = torch.nn.ModuleList()
        for model_id, model in ms_models.items():
            self.ms_models.append(instantiate(model["init_params"], _recursive_=False))

        self.input_size = [model.input_size for model in self.ms_models]

        self.ms_output_sizes = [model.output_size for model in self.ms_models]

        if shared_net["name"].startswith("TabNet"):
            param_dict = dict( cat_idxs = [], cat_dims = [], input_dim= sum(self.input_size) )
        elif shared_net["name"].startswith(("TabTransformer", "FTTransformer")):
            param_dict = dict( cat_idxs = [], categories = [], num_continuous = sum(self.ms_output_sizes), embed_input= False )
        elif shared_net["name"].startswith("naim"):
            d_token = self.ms_models[0].d_token
            # ms_output_size = int(sum(self.ms_output_sizes) / d_token)
            param_dict = dict(embed_input= False, d_token= d_token)
        elif shared_net["name"].startswith("MLP"):
            param_dict = dict(input_size= sum(self.ms_output_sizes))
        else:
            param_dict = dict()

        shared_net = recursive_cfg_substitute(shared_net, param_dict)
        self.shared_net = instantiate(shared_net["init_params"], _recursive_=False)

        self.output_size = self.shared_net.output_size

    def forward(self, *multiple_inputs):
        hidden_representations = list()
        for inputs, model in zip(multiple_inputs, self.ms_models):
            hidden_representations.append(model(inputs))

        hidden_representations = torch.cat(hidden_representations, dim=1)

        out = self.shared_net(hidden_representations, multiple_inputs)

        return out


if __name__ == "__main__":
    pass
