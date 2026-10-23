from transformers import AutoModel, AutoTokenizer
from .base import EpigenomicsModel
from torch import nn


def tokenizer_func(sequences, repo: str = "InstaDeepAI/NTv3_650M_post"):
    repo = "/storage/zhangkaiLab/libocheng/project/DECODE/data/model_paras/NTv3_650M_post"
    if repo == "InstaDeepAI/NTv3_650M_post":
        tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=True, local_files_only=True)
    inputs = tokenizer(sequences, add_special_tokens=False, padding=True, pad_to_multiple_of=128, return_tensors="pt")
    return inputs.to("cuda")


class NTv3PostTrunk(EpigenomicsModel):
    def __init__(self, repo: str | None = "InstaDeepAI/NTv3_650M_post"):
        super().__init__()
        repo = "/storage/zhangkaiLab/libocheng/project/DECODE/data/model_paras/NTv3_650M_post"
        if repo == "InstaDeepAI/NTv3_650M_post":
            self._model = AutoModel.from_pretrained(repo, trust_remote_code=True)
        else:
            self._model = AutoModel.from_pretrained(repo, trust_remote_code=True, local_files_only=True)
        self._output_dim = 1536
        self._input_length = 524288
        self._output_length = 524288
        self._output_resolution = 1
        species = ['human']
        self.species_ids = self._model.encode_species(species).to('cuda')
        self._model.eval()

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
        outputs = self._model(input_ids=inputs["input_ids"], species_ids=self.species_ids)
        return outputs['embedding']

    @classmethod
    def from_pretrained(cls, repo: str | None = "InstaDeepAI/NTv3_650M_post") -> "NTv3PostTrunk":
        return NTv3PostTrunk(repo=repo)