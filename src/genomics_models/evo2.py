import torch
from evo2 import Evo2
from .base import EpigenomicsModel
from vortex.model.tokenizer import CharLevelTokenizer
from torch import nn

def tokenizer_func(sequences):
    tokenizer = CharLevelTokenizer(512)
    return torch.tensor(
        tokenizer.tokenize(sequences),
        dtype=torch.int,
    ).unsqueeze(0)


class Evo2_Trunk(EpigenomicsModel): 
    def __init__(self, repo: str | None = None):
        super().__init__()
        if repo is not None:
            self._model = Evo2('evo2_7b', local_path=repo)
        else:
            self._model = Evo2('evo2_7b')
        self._input_length = 100000 // 2
        self._output_length = 100000 // 2
        self._output_resolution = 1
        self._output_dim = 4096
    
    @property
    def base_model(self) -> nn.Module:
        return self._model

    @property
    def input_length(self) -> int:
        return self._input_length

    @property
    def output_length(self) -> int:
        return self._output_length

    @property
    def output_resolution(self) -> int:
        return self._output_resolution

    @property
    def output_dim(self) -> int:
        return self._output_dim
    
    def forward(self, inputs):
        layer_name = 'blocks.28.mlp.l3'
        outputs, embeddings = self._model(inputs, return_embeddings=True, layer_names=[layer_name])
        return embeddings[layer_name]

    @classmethod
    def from_pretrained(cls, repo: str | None = None) -> "Evo2_Trunk":
        return Evo2_Trunk(repo)
