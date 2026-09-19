from dataclasses import dataclass

import mlx.nn as nn

from .attnres import DepthMixer, run_with_attnres
from .layers import GatedMLA, KimiDeltaAttention, LatentMoE, SiTUGLU


@dataclass
class Config:
    vocab: int = 64
    dim: int = 64
    heads: int = 4
    head_dim: int = 16
    pattern: str = "KKKA"  # K = delta attention, A = global attention; repeated `repeats` times
    repeats: int = 1
    final_global: bool = True  # end the stack with one more global-attention layer
    mla_latent: int = 32
    chunk: int = 16
    use_moe: bool = True
    latent: int = 32
    n_routed: int = 16
    top_k: int = 2
    routed_hidden: int = 32
    n_shared: int = 2
    shared_hidden: int = 64
    dense_hidden: int = 128
    attn_res: bool = True
    res_block: int = 4  # sublayers per attention-residual block


class Sublayer(nn.Module):
    """Pre-norm wrapper returning only the layer's contribution, never the residual."""

    def __init__(self, dim, body, takes_collect=False):
        super().__init__()
        self.norm = nn.RMSNorm(dim)
        self.body = body
        self.takes_collect = takes_collect

    def __call__(self, h, collect=None):
        if self.takes_collect:
            return self.body(self.norm(h), collect)
        return self.body(self.norm(h))


def build_layers(cfg):
    kinds = list(cfg.pattern * cfg.repeats) + (["A"] if cfg.final_global else [])
    layers = []
    for kind in kinds:
        if kind == "K":
            mixer = KimiDeltaAttention(cfg.dim, cfg.heads, cfg.head_dim, chunk=cfg.chunk)
        elif kind == "A":
            mixer = GatedMLA(cfg.dim, cfg.heads, cfg.head_dim, cfg.mla_latent)
        else:
            raise ValueError(f"unknown layer kind {kind!r}")
        layers.append(Sublayer(cfg.dim, mixer))
        if cfg.use_moe:
            ffn = LatentMoE(
                cfg.dim, cfg.latent, cfg.n_routed, cfg.top_k, cfg.routed_hidden, cfg.n_shared, cfg.shared_hidden
            )
            layers.append(Sublayer(cfg.dim, ffn, takes_collect=True))
        else:
            layers.append(Sublayer(cfg.dim, SiTUGLU(cfg.dim, cfg.dense_hidden)))
    return layers, kinds


class DeltaLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab, cfg.dim)
        self.layers, self.kinds = build_layers(cfg)
        self.mixer = DepthMixer(cfg.dim, len(self.layers))
        self.norm = nn.RMSNorm(cfg.dim)
        self.head = nn.Linear(cfg.dim, cfg.vocab, bias=False)

    def moes(self):
        return [l.body for l in self.layers if isinstance(l.body, LatentMoE)]

    def __call__(self, tokens, collect=None):
        """`collect`, if given, receives each MoE layer's refreshed routing bias."""
        x = self.embed(tokens)
        if self.cfg.attn_res:
            x = run_with_attnres(self.mixer, self.layers, x, self.cfg.res_block, collect)
        else:
            for layer in self.layers:
                x = x + layer(x, collect)
        return self.head(self.norm(x))

    def apply_bias_updates(self, biases):
        for moe, bias in zip(self.moes(), biases):
            moe.bias = bias
