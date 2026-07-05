import torch
from typing import Union
from omegaconf import ListConfig
from torch.nn import LeakyReLU, ReLU, Sigmoid, Tanh, Softmax

__all__ = ["CustomMLP"]


class CustomMLP(torch.nn.Module):
    """
    Custom MLP class
    """
    def __init__(self, input_size: Union[int, tuple], output_size: int, hidden_sizes: list = list, activation_functions: Union[ list, str ] = "relu", drop_rate: float = None, extractor: bool = False):

        super(CustomMLP, self).__init__()

        if type(activation_functions) is str and hidden_sizes is not None:
            activation_functions = [activation_functions] * len(hidden_sizes)  # (len(hidden_sizes) + 1)

        activation_options = dict(leakyrelu=LeakyReLU, relu=ReLU, sigmoid=Sigmoid, tanh=Tanh, softmax=Softmax)
        activation_params = dict(leakyrelu=dict(negative_slope=1e-2), relu={}, sigmoid={}, tanh={}, softmax=dict(dim=1))

        self.extractor = extractor
        # Input layer
        if isinstance(input_size, (tuple, ListConfig)):
            input_size = input_size[0]

        if drop_rate is None:  #
            self.input = torch.nn.Sequential(torch.nn.Linear(input_size, hidden_sizes[0]), activation_options[activation_functions[0]](**activation_params[activation_functions[0]]))
        else:
            self.input = torch.nn.Sequential(torch.nn.Linear(input_size, hidden_sizes[0]), activation_options[activation_functions[0]](**activation_params[activation_functions[0]]), torch.nn.Dropout(drop_rate))

        # Hidden layers
        self.hidden = torch.nn.ModuleList()
        if len(hidden_sizes) > 1:
            for k in range(len(hidden_sizes) - 1):
                if drop_rate is not None and k < len(hidden_sizes) - 2:
                    layer = torch.nn.Sequential(torch.nn.Linear(hidden_sizes[k], hidden_sizes[k + 1]), activation_options[activation_functions[k+1]](**activation_params[activation_functions[k+1]]), torch.nn.Dropout(drop_rate))
                else:
                    layer = torch.nn.Sequential(torch.nn.Linear(hidden_sizes[k], hidden_sizes[k + 1]), activation_options[activation_functions[k+1]](**activation_params[activation_functions[k+1]]))

                self.hidden.append(layer)

        self.input_size = input_size
        if self.extractor:
            self.output_size = hidden_sizes[-1]
        elif isinstance(output_size, (tuple, ListConfig)):
            self.output_size = output_size
        else:
            self.output_size = output_size

        # Output layer
        if not self.extractor:
            # output_activation_options = {False: Sigmoid(), True: Softmax(dim=1)}
            if self.output_size > 1:
                self.output = torch.nn.Sequential(torch.nn.Linear(hidden_sizes[-1], output_size))  # , output_activation_options[output_size > 1])
            else:
                self.output = torch.nn.Sequential(torch.nn.Linear(hidden_sizes[-1], output_size), Sigmoid())  # , output_activation_options[output_size > 1])

    def forward(self, inputs, *_):

        # Feedforward
        x = self.input(inputs)

        if self.hidden:
            for layer in self.hidden:
                x = layer(x)

        if self.extractor:
            return x

        outputs = self.output(x)

        return outputs


if __name__ == "__main__":
    pass
