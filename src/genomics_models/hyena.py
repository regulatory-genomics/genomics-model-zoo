import torch
from .base import EpigenomicsModel
from torch import nn
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union
from transformers import AutoTokenizer, AutoModel


def tokenizer_func(sequences):
    tokenizer_hyena = AutoTokenizer.from_pretrained(
        "LongSafari/hyenadna-large-1m-seqlen-hf",
        trust_remote_code=True
    )
    inputs = tokenizer_hyena(sequences, return_tensors="pt", truncation=True, max_length=1000002)
    return inputs


class hyenaTrunk(EpigenomicsModel): 
    def __init__(self, repo: str | None = None):
        super().__init__()
        if repo is None:
            self._model = AutoModel.from_pretrained(
                "LongSafari/hyenadna-large-1m-seqlen-hf",
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            self._model = AutoModel.from_pretrained(
                repo,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
        self._input_length = 196608
        self._output_length = 196608
        self._output_resolution = 1
        self._output_dim = 256
    
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
    
    def forward(self, input_ids):
        outputs = self._model(**input_ids, output_hidden_states=True)
        embedding = outputs.hidden_states[-1][:, :-1, :]
        return embedding

    @classmethod
    def from_pretrained(cls, repo: str | None = None) -> "hyenaTrunk":
        return hyenaTrunk(repo)
