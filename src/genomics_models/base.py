from torch import nn
from abc import ABC, abstractmethod
from pathlib import Path

class EpigenomicsModel(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @property
    @abstractmethod
    def base_model(self) -> nn.Module:
        """Returns the base model architecture."""
        pass

    @property
    @abstractmethod
    def input_length(self) -> int:
        """Returns the expected input sequence length."""
        pass

    @property
    @abstractmethod
    def output_length(self) -> int:
        """Returns the expected output sequence length."""
        pass

    @property
    @abstractmethod
    def output_resolution(self) -> int:
        """Returns the output resolution."""
        pass

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Returns the output dimension."""
        pass

    @classmethod
    @abstractmethod
    def from_pretrained(cls, path: Path | None = None) -> 'EpigenomicsModel':
        """Loads a pretrained model from the specified path."""
        pass