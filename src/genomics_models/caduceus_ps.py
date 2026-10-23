import torch
from .base import EpigenomicsModel
from torch import nn
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union
from transformers import AutoTokenizer, AutoModel


def tokenizer_func(sequences):
    model_name_ = "kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16"
    tokenizer_caduceus = AutoTokenizer.from_pretrained(model_name_, trust_remote_code=True)
    inputs = tokenizer_caduceus(sequences, return_tensors="pt", truncation=True, max_length=131072+1)
    return inputs


class caducesTrunk(EpigenomicsModel): 
    def __init__(self, repo: str | None = None):
        super().__init__()
        if repo is not None:
            self._model = AutoModel.from_pretrained(
                repo, 
                trust_remote_code=True, 
                torch_dtype=torch.bfloat16, 
                device_map="auto"
            )
        else:
            self._model = AutoModel.from_pretrained(
                "kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16",
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
        self._input_length = 131072
        self._output_length = 131072
        self._output_resolution = 1
        self._output_dim = 512
    
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
        outputs = self._model(**inputs, output_hidden_states=True)
        embedding = outputs.hidden_states[-1][:, :-1, :]
        return embedding

    @classmethod
    def from_pretrained(cls, repo: str | None = None) -> "caducesTrunk":
        return caducesTrunk(repo)
