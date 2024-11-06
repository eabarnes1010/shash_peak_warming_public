"""Base model modules for pytorch models.

Classes
---------
BaseModel(torch.nn.Module)

"""

import torch
import numpy as np
from abc import abstractmethod


class BaseModel(torch.nn.Module):
    """
    Base class for all models.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, *inputs):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def predict(self, *inputs):
        """
        Prediction loop.

        :return: Model output
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + "\nTrainable parameters: {}".format(params)

    def freeze_layers(self, freeze_id, verbose=False):
        params = self.state_dict()
        params.keys()

        for name, param in self.named_parameters():
            if freeze_id in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

        if verbose:
            for name, param in self.named_parameters():
                print("-" * 20)
                print(f"name: {name}, ")
                print(str(param.numel()))
                print(", train: ")
                print(param.requires_grad)
