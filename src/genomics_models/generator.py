import torch
from .base import EpigenomicsModel
from torch import nn
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union
from transformers import AutoTokenizer, AutoModelForCausalLM



def tokenizer_func(sequences):
    tokenizer = AutoTokenizer.from_pretrained("GenerTeam/GENERator-v2-eukaryote-3b-base", trust_remote_code=True)
    if isinstance(sequences, str):
        sequences = [sequences]
    processed_sequences = [tokenizer.bos_token + seq[:len(seq)//6*6] for seq in sequences]
    max_length=16384
    return tokenizer(
        processed_sequences,
        add_special_tokens=True,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length
    )



class generatorTrunk(EpigenomicsModel): 
    def __init__(self, repo: str | None = None):
        super().__init__()
        if repo is not None:
            self._model = AutoModelForCausalLM.from_pretrained(repo)
        else:
            self._model = AutoModelForCausalLM.from_pretrained("GenerTeam/GENERator-v2-eukaryote-3b-base")
        self._input_length = 98304
        self._output_length = 98304 // 6
        self._output_resolution = 6
        self._output_dim = 3072
    
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
        hidden_states = outputs.hidden_states[-1]
        attention_mask = inputs["attention_mask"]
        expanded_mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).to(torch.float32)
        tmp = hidden_states * expanded_mask
        return tmp[:, 2:-1,:]

    @classmethod
    def from_pretrained(cls, repo: str | None = None) -> "generatorTrunk":
        return generatorTrunk(repo)
