import abc
import torch
from torch import nn

"""
Generic loss class.
Each must define the domain ('feature' or 'time') it is applied in as well as a name.
Child classes do not need to make use of the .weight attribute themselves, this should be handled by the model/trainer/caller impl.
Implementations should generally sum over all dimensions. Turning this into a batchwise mean is the responsibility of the caller.

Implementations should receive inputs (target, estimate) and return a loss value.
"""
class Loss(abc.ABC, nn.Module):
    def __init__(self, *args, weight=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self._weight = weight

    @property
    @abc.abstractmethod
    def domain(self):
        pass

    @property
    @abc.abstractmethod
    def name(self):
        pass

    @abc.abstractmethod
    def forward(self, x: torch.Tensor, xhat: torch.Tensor):
        pass

    @property
    def weight(self):
        return self._weight
