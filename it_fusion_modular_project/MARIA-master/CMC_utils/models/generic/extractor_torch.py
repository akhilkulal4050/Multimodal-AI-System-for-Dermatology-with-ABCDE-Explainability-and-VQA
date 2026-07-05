import torch
from CMC_utils.models import list_model_modules

__all__ = ["Extractor"]

class Extractor(torch.nn.Module):
    """
    Extractor class, used to extract features from a model
    """
    def __init__(self, model, *num_ftrs, **kwargs):
        super(Extractor, self).__init__()

        self.feature_extractor = torch.nn.Sequential(*list_model_modules(model))  # torch.nn.Sequential(*list(model.children()))

        output_size = kwargs.get("output_size", None)

        self.output_layer = output_size is not None

        if output_size:
            self.fc = torch.nn.Linear(in_features=sum(num_ftrs), out_features=output_size)
            self.output_size = [output_size]

        else:
            self.output_size = list(num_ftrs)

    def forward(self, inputs):

        x = self.feature_extractor(inputs)

        if self.output_layer:
            x = self.fc(x.squeeze())

        return x


if __name__ == "__main__":
    pass
