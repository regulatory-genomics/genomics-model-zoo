# Copyright 2023 Calico LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================


import torch
from torch import nn, einsum
from torch.amp import autocast
import math
import copy
from einops import rearrange
from pathlib import Path
from dataclasses import dataclass

from .utils import seq_indices_to_one_hot


class BorzoiTrunk(nn.Module):
    def __init__(self, path: Path | None = None):
        super().__init__()
        self.borzoi = fetch_pretrained(path)
        self.input_length = 524_288
        self.output_length = 196_608
        self.output_resolution = 32
        self.emb_dim = 1920

    def forward(self, x):
        if not torch.is_floating_point(x):
            x = seq_indices_to_one_hot(x)

        x = x.permute(0, 2, 1)  # (B, 4, L)
        with autocast(device_type='cuda', dtype=torch.bfloat16):
            x = self.borzoi.get_embs_after_crop(x)
            x = self.borzoi.final_joined_convs(x)
        x = x.permute(0, 2, 1)  # (B, L, C)
        return x


def fetch_pretrained(path: Path | None = None) -> "Borzoi":
    from pooch import retrieve

    config = BorzoiConfig()
    model = Borzoi(config)
    if path is None:
        path = retrieve(
            "https://www.modelscope.cn/models/regulatory-genomics-lab/Borzoi/resolve/master/pytorch_model.bin",
            known_hash="sha256:16dbb77387b2531d87669cc5d61b14a9918d48835be768aaa5b5d9a13cde7ec3",
            fname="borzoi.bin",
            progressbar=True,
        )
    if torch.cuda.is_available():
        state_dict = torch.load(path)
    else:
        state_dict = torch.load(path, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    return model


@dataclass
class BorzoiConfig:
    dim: int = 1536
    depth: int = 8
    heads: int = 8
    return_center_bins_only: bool = True
    attn_dim_key: int = 64
    attn_dim_value: int = 192
    dropout_rate: float = 0.2
    attn_dropout: float = 0.05
    pos_dropout: float = 0.01
    enable_mouse_head: bool = True
    bins_to_return: int = 6144
    flashed: bool = True


class Borzoi(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.flashed = config.flashed if "flashed" in config.__dict__.keys() else False
        self.enable_human_head = (
            config.enable_human_head
            if "enable_human_head" in config.__dict__.keys()
            else True
        )
        self.enable_mouse_head = config.enable_mouse_head
        self.conv_dna = ConvDna()
        self._max_pool = nn.MaxPool1d(kernel_size=2, padding=0)
        self.res_tower = nn.Sequential(
            ConvBlock(in_channels=512, out_channels=608, kernel_size=5),
            self._max_pool,
            ConvBlock(in_channels=608, out_channels=736, kernel_size=5),
            self._max_pool,
            ConvBlock(in_channels=736, out_channels=896, kernel_size=5),
            self._max_pool,
            ConvBlock(in_channels=896, out_channels=1056, kernel_size=5),
            self._max_pool,
            ConvBlock(in_channels=1056, out_channels=1280, kernel_size=5),
        )
        self.unet1 = nn.Sequential(
            self._max_pool,
            ConvBlock(in_channels=1280, out_channels=config.dim, kernel_size=5),
        )
        transformer = []
        for _ in range(config.depth):
            transformer.append(
                nn.Sequential(
                    Residual(
                        nn.Sequential(
                            nn.LayerNorm(config.dim, eps=0.001),
                            Attention(
                                config.dim,
                                heads=config.heads,
                                dim_key=config.attn_dim_key,
                                dim_value=config.attn_dim_value,
                                dropout=config.attn_dropout,
                                pos_dropout=config.pos_dropout,
                                num_rel_pos_features=32,
                            )
                            if not self.flashed
                            else FlashAttention(
                                config.dim,
                                heads=config.heads,
                                dropout=config.attn_dropout,
                                pos_dropout=config.pos_dropout,
                            ),
                            nn.Dropout(0.2),
                        )
                    ),
                    Residual(
                        nn.Sequential(
                            nn.LayerNorm(config.dim, eps=0.001),
                            nn.Linear(config.dim, config.dim * 2),
                            nn.Dropout(config.dropout_rate),
                            nn.ReLU(),
                            nn.Linear(config.dim * 2, config.dim),
                            nn.Dropout(config.dropout_rate),
                        )
                    ),
                )
            )
        self.horizontal_conv0, self.horizontal_conv1 = (
            ConvBlock(in_channels=1280, out_channels=config.dim, kernel_size=1),
            ConvBlock(in_channels=config.dim, out_channels=config.dim, kernel_size=1),
        )
        self.upsample = torch.nn.Upsample(scale_factor=2)
        self.transformer = nn.Sequential(*transformer)
        self.upsampling_unet1 = nn.Sequential(
            ConvBlock(in_channels=config.dim, out_channels=config.dim, kernel_size=1),
            self.upsample,
        )
        self.separable1 = ConvBlock(
            in_channels=config.dim,
            out_channels=config.dim,
            kernel_size=3,
            conv_type="separable",
        )
        self.upsampling_unet0 = nn.Sequential(
            ConvBlock(in_channels=config.dim, out_channels=config.dim, kernel_size=1),
            self.upsample,
        )
        self.separable0 = ConvBlock(
            in_channels=config.dim,
            out_channels=config.dim,
            kernel_size=3,
            conv_type="separable",
        )
        if config.return_center_bins_only:
            self.crop = TargetLengthCrop(config.bins_to_return)
        else:
            self.crop = TargetLengthCrop(16384 - 32)  # as in Borzoi
        self.final_joined_convs = nn.Sequential(
            ConvBlock(in_channels=config.dim, out_channels=1920, kernel_size=1),
            nn.Dropout(0.1),
            nn.GELU(approximate="tanh"),
        )
        if self.enable_human_head:
            self.human_head = nn.Conv1d(
                in_channels=1920, out_channels=7611, kernel_size=1
            )
        if self.enable_mouse_head:
            self.mouse_head = nn.Conv1d(
                in_channels=1920, out_channels=2608, kernel_size=1
            )
        self.final_softplus = nn.Softplus()

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
            # module.weight.data.normal_(mean=0.0, std=0.02)
            nn.init.xavier_normal_(module.weight)
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, (nn.Linear, nn.Conv1d)) and module.bias is not None:
            module.bias.data.zero_()

    def set_track_subset(self, track_subset):
        """
        Creates a subset of tracks by reassigning weights in the human head.

        Args:
           track_subset: Indices of the tracks to keep.

        Returns:
            None
        """
        if not hasattr(self, "human_head_bak"):
            self.human_head_bak = copy.deepcopy(self.human_head)
        else:
            self.reset_track_subset()
        self.human_head = nn.Conv1d(1920, len(track_subset), 1)
        self.human_head.weight = nn.Parameter(
            self.human_head_bak.weight[track_subset].clone()
        )
        self.human_head.bias = nn.Parameter(
            self.human_head_bak.bias[track_subset].clone()
        )

    def reset_track_subset(self):
        """
        Resets the human head to the original weights.

        Returns:
            None
        """
        self.human_head = copy.deepcopy(self.human_head_bak)

    def get_embs_after_crop(self, x):
        """
        Performs the forward pass of the model until right before the final conv layers, and includes a cropping layer.

        Args:
            x (torch.Tensor): Input DNA sequence tensor of shape (N, 4, L).

        Returns:
             torch.Tensor: Output of the model up to the cropping layer with shape (N, dim, crop_length)
        """
        x = self.conv_dna(x)
        x_unet0 = self.res_tower(x)
        x_unet1 = self.unet1(x_unet0)
        x = self._max_pool(x_unet1)
        x_unet1 = self.horizontal_conv1(x_unet1)
        x_unet0 = self.horizontal_conv0(x_unet0)
        x = self.transformer(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        x = self.upsampling_unet1(x)
        x += x_unet1
        x = self.separable1(x)
        x = self.upsampling_unet0(x)
        x += x_unet0
        x = self.separable0(x)
        x = self.crop(x.permute(0, 2, 1))
        return x.permute(0, 2, 1)

    def predict(self, seqs, gene_slices, remove_squashed_scale=False):
        """
        Predicts only for bins of interest in a batched fashion
        Args:
            seqs (torch.tensor): Nx4xL tensor of one-hot sequences
            gene_slices List[torch.Tensor]: tensors indicating bins of interest
            removed_squashed_scale (bool, optional): whether to undo the squashed scale

        Returns:
            Tuple[torch.Tensor, list[int]]: 1xCxB tensor of bin predictions, as well as offsets that indicate where sequences begin/end
        """
        # Calculate slice offsets
        slice_list = []
        slice_length = []
        offset = self.crop.target_length
        for i, gene_slice in enumerate(gene_slices):
            slice_list.append(gene_slice + i * offset)
            slice_length.append(gene_slice.shape[0])
        slice_list = torch.concatenate(slice_list)
        # Get embedding after cropped
        seq_embs = self.get_embs_after_crop(seqs)
        # Reshape to flatten the batch dimension (i.e. concatenate sequences)
        seq_embs = seq_embs.permute(1, 0, 2).flatten(start_dim=1).unsqueeze(0)
        # Extract the bins of interest
        seq_embs = seq_embs[:, :, slice_list]
        # Run the model head
        seq_embs = self.final_joined_convs(seq_embs)
        with torch.amp.autocast("cuda", enabled=False):
            conved_slices = self.final_softplus(self.human_head(seq_embs.float()))
        if remove_squashed_scale:
            conved_slices = undo_squashed_scale(conved_slices)
        return conved_slices, slice_length

    def forward(self, x, is_human=True, data_parallel_training=False):
        """
        Performs the forward pass of the model.

        Args:
            x (torch.Tensor): Input DNA sequence tensor of shape (N, 4, L).
            is_human (bool, optional): If True, use the human head; otherwise, use the mouse head. Defaults to True.
            data_parallel_training (bool, optional): If True, perform forward pass specific to DDP. Defaults to False.

        Returns:
            torch.Tensor: Output tensor with shape (N, C, L), where C is the number of tracks.
        """
        x = self.get_embs_after_crop(x)
        x = self.final_joined_convs(x)
        # disable autocast for more precision in final layer
        with torch.amp.autocast("cuda", enabled=False):
            if data_parallel_training:
                # we need this to get gradients for both heads if doing DDP training
                if is_human:
                    human_out = (
                        self.final_softplus(self.human_head(x.float()))
                        + 0 * self.mouse_head(x.float()).sum()
                    )
                    return human_out
                else:
                    mouse_out = (
                        self.final_softplus(self.mouse_head(x.float()))
                        + 0 * self.human_head(x.float()).sum()
                    )
                    return mouse_out
            else:
                if is_human:
                    return self.final_softplus(self.human_head(x.float()))
                else:
                    return self.final_softplus(self.mouse_head(x.float()))


class ConvDna(nn.Module):
    def __init__(self):
        super(ConvDna, self).__init__()
        self.conv_layer = nn.Conv1d(
            in_channels=4, out_channels=512, kernel_size=15, padding="same"
        )
        self.max_pool = nn.MaxPool1d(kernel_size=2, padding=0)

    def forward(self, x):
        return self.max_pool(self.conv_layer(x))


class ConvBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels=None, kernel_size=1, conv_type="standard"
    ):
        super(ConvBlock, self).__init__()
        if conv_type == "separable":
            self.norm = nn.Identity()
            depthwise_conv = nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                groups=in_channels,
                padding="same",
                bias=False,
            )
            pointwise_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
            self.conv_layer = nn.Sequential(depthwise_conv, pointwise_conv)
            self.activation = nn.Identity()
        else:
            self.norm = nn.BatchNorm1d(in_channels, eps=0.001)
            self.activation = nn.GELU(approximate="tanh")
            self.conv_layer = nn.Conv1d(
                in_channels, out_channels, kernel_size=kernel_size, padding="same"
            )

    def forward(self, x):
        x = self.norm(x)
        x = self.activation(x)
        x = self.conv_layer(x)
        return x


def get_positional_features_central_mask(positions, features, seq_len):
    pow_rate = math.exp(math.log(seq_len + 1) / features)
    center_widths = torch.pow(
        pow_rate, torch.arange(1, features + 1, device=positions.device)
    ).float()
    center_widths = center_widths - 1
    return (center_widths[None, ...] > positions.abs()[..., None]).float()


def get_positional_embed(seq_len, feature_size, device):
    distances = torch.arange(-seq_len + 1, seq_len, device=device)

    feature_functions = [
        get_positional_features_central_mask,
    ]

    num_components = len(feature_functions) * 2

    if (feature_size % num_components) != 0:
        raise ValueError(
            f"feature size is not divisible by number of components ({num_components})"
        )

    num_basis_per_class = feature_size // num_components

    embeddings = []
    for fn in feature_functions:
        embeddings.append(fn(distances, num_basis_per_class, seq_len))

    embeddings = torch.cat(embeddings, dim=-1)
    embeddings = torch.cat(
        (embeddings, torch.sign(distances)[..., None] * embeddings), dim=-1
    )
    return embeddings


def fast_relative_shift(a, b):
    return (
        einsum("i d, j d -> i j", a, b)
        .flatten()
        .as_strided(
            size=(a.shape[0], a.shape[0]),
            stride=((a.shape[0] - 1) * 2, 1),
            storage_offset=a.shape[0] - 1,
        )
    )


fast_relative_shift = torch.vmap(
    torch.vmap(fast_relative_shift), in_dims=(0, None)
)  # https://johahi.github.io/blog/2024/fast-relative-shift/


class Attention(nn.Module):
    def __init__(
        self,
        dim=1536,
        *,
        num_rel_pos_features=1,
        heads=8,
        dim_key=64,
        dim_value=64,
        dropout=0.0,
        pos_dropout=0.0,
    ):
        super().__init__()
        self.scale = dim_key**-0.5
        self.heads = heads

        self.to_q = nn.Linear(dim, dim_key * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_key * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_value * heads, bias=False)

        self.to_out = nn.Linear(dim_value * heads, dim)
        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

        # relative positional encoding

        self.num_rel_pos_features = num_rel_pos_features

        self.register_buffer(
            "positions",
            get_positional_embed(
                4096, self.num_rel_pos_features, self.to_v.weight.device
            ),
            persistent=False,
        )  # 4096 as this should always be the seq len at this pos?

        self.to_rel_k = nn.Linear(num_rel_pos_features, dim_key * heads, bias=False)
        self.rel_content_bias = nn.Parameter(torch.randn(1, heads, 1, dim_key))
        self.rel_pos_bias = nn.Parameter(torch.randn(1, heads, 1, dim_key))

        # dropouts

        self.pos_dropout = nn.Dropout(pos_dropout)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x):
        n, h, device = x.shape[-2], self.heads, x.device

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

        q = q * self.scale

        content_logits = einsum(
            "b h i d, b h j d -> b h i j", q + self.rel_content_bias, k
        )

        positions = self.pos_dropout(self.positions)
        rel_k = self.to_rel_k(positions)
        rel_k = rearrange(rel_k, "n (h d) -> h n d", h=h)
        rel_logits = fast_relative_shift(q + self.rel_pos_bias, rel_k)
        logits = content_logits + rel_logits
        attn = logits.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)
        return out


class FlashAttention(nn.Module):
    def __init__(
        self,
        dim=1536,
        heads=8,
        dropout=0.15,
        pos_dropout=0.15,  # Not used
        rotary_emb_base=20000.0,
        rotary_emb_scale_base=None,
    ):
        super().__init__()

        from flash_attn.modules.mha import MHA

        self.mha = MHA(
            use_flash_attn=True,
            embed_dim=dim,
            num_heads=heads,
            num_heads_kv=(heads // 2),
            qkv_proj_bias=True,  # False,
            out_proj_bias=True,
            dropout=dropout,
            softmax_scale=(dim / heads) ** -0.5,
            causal=False,
            rotary_emb_dim=128,
            rotary_emb_base=rotary_emb_base,
            rotary_emb_scale_base=rotary_emb_scale_base,
            fused_bias_fc=False,
        )

        nn.init.kaiming_normal_(self.mha.Wqkv.weight, nonlinearity="relu")
        nn.init.zeros_(self.mha.out_proj.weight)
        nn.init.zeros_(self.mha.out_proj.bias)
        nn.init.ones_(self.mha.Wqkv.bias)

    def forward(self, x):
        out = self.mha(x)
        return out


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class TargetLengthCrop(nn.Module):
    def __init__(self, target_length):
        super().__init__()
        self.target_length = target_length

    def forward(self, x):
        seq_len, target_len = x.shape[-2], self.target_length

        if target_len == -1:
            return x

        if seq_len < target_len:
            raise ValueError(
                f"sequence length {seq_len} is less than target length {target_len}"
            )

        trim = (target_len - seq_len) // 2

        if trim == 0:
            return x

        return x[:, -trim:trim]


def undo_squashed_scale(
    x, clip_soft=384, track_transform=3 / 4, track_scale=0.01, old_transform=True
):
    """
    Reverses the squashed scaling transformation applied to the output profiles.

    Args:
        x (torch.Tensor): The input tensor to be unsquashed.
        clip_soft (float, optional): The soft clipping value. Defaults to 384.
        track_transform (float, optional): The transformation factor. Defaults to 3/4.
        track_scale (float, optional): The scale factor. Defaults to 0.01.

    Returns:
        torch.Tensor: The unsquashed tensor.
    """
    x = x.clone()  # IMPORTANT BECAUSE OF IMPLACE OPERATIONS TO FOLLOW?

    if old_transform:
        x = x / track_scale
        unclip_mask = x > clip_soft
        x[unclip_mask] = (x[unclip_mask] - clip_soft) ** 2 + clip_soft
        x = x ** (1.0 / track_transform)
    else:
        unclip_mask = x > clip_soft
        x[unclip_mask] = (x[unclip_mask] - clip_soft + 1) ** 2 + clip_soft - 1
        x = (x + 1) ** (1.0 / track_transform) - 1
        x = x / track_scale
    return x
