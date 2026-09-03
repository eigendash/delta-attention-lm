"""Token-mixing and channel-mixing layers for the hybrid model."""

import math

import mlx.core as mx
import mlx.nn as nn

from .kda import chunk_kda, recurrent_kda


def l2norm(x, eps=1e-6):
    return x * mx.rsqrt((x * x).sum(axis=-1, keepdims=True) + eps)


def softcap(x, bound):
    return bound * mx.tanh(x / bound)


class ShortConv(nn.Module):
    """Causal depthwise convolution over time on (B, T, C)."""

    def __init__(self, channels, width=4):
        super().__init__()
        self.width = width
        self.conv = nn.Conv1d(channels, channels, width, padding=0, groups=channels, bias=False)

    def __call__(self, x):
        return self.conv(mx.pad(x, [(0, 0), (self.width - 1, 0), (0, 0)]))


class KimiDeltaAttention(nn.Module):
    """Gated delta-rule attention with a lower-bounded, channel-wise decay.

    The per-step log decay is g = g_min * sigmoid(exp(A_h) * z). Bounding it keeps
    the reciprocal cumulative decay finite inside a chunk, which is what makes the
    chunkwise form numerically safe.
    """

    def __init__(self, dim, heads, head_dim, g_min=-5.0, chunk=16, decay_rank=16, impl="chunk"):
        super().__init__()
        inner = heads * head_dim
        self.heads, self.head_dim, self.g_min, self.chunk, self.impl = heads, head_dim, g_min, chunk, impl
        self.q_proj = nn.Linear(dim, inner, bias=False)
        self.k_proj = nn.Linear(dim, inner, bias=False)
        self.v_proj = nn.Linear(dim, inner, bias=False)
        self.q_conv, self.k_conv, self.v_conv = (ShortConv(inner) for _ in range(3))
        self.beta_proj = nn.Linear(dim, heads, bias=False)
        self.decay_down = nn.Linear(dim, decay_rank, bias=False)
        self.decay_up = nn.Linear(decay_rank, inner, bias=True)
        self.log_scale = mx.zeros((heads,))
        self.gate_proj = nn.Linear(dim, inner, bias=False)
        self.out_norm = nn.RMSNorm(head_dim)
        self.o_proj = nn.Linear(inner, dim, bias=False)
        self._init_decay_bias(inner)

    def _init_decay_bias(self, inner):
        # Spread initial retention from fast-forgetting to nearly-permanent channels.
        target = -mx.exp(mx.linspace(math.log(0.01), math.log(1.0), inner))
        frac = mx.clip(target / self.g_min, 1e-4, 1 - 1e-4)
        self.decay_up.bias = mx.log(frac / (1 - frac))

    def __call__(self, x):
        B, T, _ = x.shape
        H, dh = self.heads, self.head_dim

        def mix(proj, conv, normalize):
            t = nn.silu(conv(proj(x))).reshape(B, T, H, dh)
            return (l2norm(t) if normalize else t).transpose(0, 2, 1, 3)

        q = mix(self.q_proj, self.q_conv, True)
        k = mix(self.k_proj, self.k_conv, True)
        v = mix(self.v_proj, self.v_conv, False)
        z = self.decay_up(self.decay_down(x)).reshape(B, T, H, dh).transpose(0, 2, 1, 3)
        log_decay = self.g_min * mx.sigmoid(mx.exp(self.log_scale)[None, :, None, None] * z)
        beta = mx.sigmoid(self.beta_proj(x)).transpose(0, 2, 1)

        fn = chunk_kda if self.impl == "chunk" else recurrent_kda
        kwargs = {"chunk": self.chunk} if self.impl == "chunk" else {}
        o, _ = fn(q, k, v, log_decay, beta, **kwargs)

        o = self.out_norm(o.transpose(0, 2, 1, 3))
        gate = mx.sigmoid(self.gate_proj(x)).reshape(B, T, H, dh)
        return self.o_proj((gate * o).reshape(B, T, H * dh))


class GatedMLA(nn.Module):
    """Global attention through a shared KV latent, no positional encoding, output gate."""

    def __init__(self, dim, heads, head_dim, latent):
        super().__init__()
        inner = heads * head_dim
        self.heads = heads
        self.q_proj = nn.Linear(dim, inner, bias=False)
        self.kv_down = nn.Linear(dim, latent, bias=False)
        self.k_up = nn.Linear(latent, inner, bias=False)
        self.v_up = nn.Linear(latent, inner, bias=False)
        self.gate_proj = nn.Linear(dim, inner, bias=False)
        self.o_proj = nn.Linear(inner, dim, bias=False)

    def __call__(self, x):
        B, T, _ = x.shape
        latent = self.kv_down(x)

        def heads_first(t):
            return t.reshape(B, T, self.heads, -1).transpose(0, 2, 1, 3)

        q, k, v = heads_first(self.q_proj(x)), heads_first(self.k_up(latent)), heads_first(self.v_up(latent))
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=q.shape[-1] ** -0.5, mask="causal")
        o = o.transpose(0, 2, 1, 3).reshape(B, T, -1)
        return self.o_proj(mx.sigmoid(self.gate_proj(x)) * o)


class SiTUGLU(nn.Module):
    """GLU whose gate and up branches are both soft-capped, so the product is bounded.

    f(x) = [b1 tanh(g/b1) * sigmoid(g)] * [b2 tanh(u/b2)]  with |f| <= b1 * b2.
    Near zero it behaves like SwiGLU.
    """

    def __init__(self, dim, hidden, bound_gate=4.0, bound_up=25.0):
        super().__init__()
        self.bound_gate, self.bound_up = bound_gate, bound_up
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.up_proj = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)

    def hidden(self, x):
        g = self.gate_proj(x)
        return softcap(g, self.bound_gate) * mx.sigmoid(g) * softcap(self.up_proj(x), self.bound_up)

    def __call__(self, x):
        return self.down_proj(self.hidden(x))


class RoutedExperts(nn.Module):
    """Experts stored as stacked tensors and evaluated for all tokens at once."""

    def __init__(self, n_experts, width, hidden, bound_gate=4.0, bound_up=25.0):
        super().__init__()
        self.bound_gate, self.bound_up = bound_gate, bound_up
        scale_in, scale_out = width**-0.5, hidden**-0.5
        self.gate = mx.random.normal((n_experts, width, hidden)) * scale_in
        self.up = mx.random.normal((n_experts, width, hidden)) * scale_in
        self.down = mx.random.normal((n_experts, hidden, width)) * scale_out

    def __call__(self, z):  # (N, width) -> (N, E, width)
        g = mx.einsum("nl,elh->neh", z, self.gate)
        u = mx.einsum("nl,elh->neh", z, self.up)
        h = softcap(g, self.bound_gate) * mx.sigmoid(g) * softcap(u, self.bound_up)
        return mx.einsum("neh,ehl->nel", h, self.down)


class LatentMoE(nn.Module):
    """Shared full-width experts plus routed experts that work in a narrow latent space.

        y = sum_j shared_j(x) + W_up RMSNorm( sum_{i in topk} p_i routed_i(W_down x) )

    Routing scores are sigmoids. A per-expert bias steers top-k selection but is left
    out of the mixing weights p, and is refreshed by quantile balancing (see
    `balanced_bias`) after each training forward pass.
    """

    def __init__(self, dim, latent, n_routed, top_k, routed_hidden, n_shared, shared_hidden):
        super().__init__()
        if top_k >= n_routed:
            raise ValueError("top_k must be smaller than n_routed")
        self.top_k = top_k
        self.shared = [SiTUGLU(dim, shared_hidden) for _ in range(n_shared)]
        self.down = nn.Linear(dim, latent, bias=False)
        self.up = nn.Linear(latent, dim, bias=False)
        self.u_norm = nn.RMSNorm(latent)
        self.router = nn.Linear(dim, n_routed, bias=False)
        self.experts = RoutedExperts(n_routed, latent, routed_hidden)
        self.bias = mx.zeros((n_routed,))
        self.freeze(keys=["bias"], recurse=False)

    def route(self, x):
        scores = mx.sigmoid(self.router(x))
        order = mx.stop_gradient(mx.argsort(-(scores + self.bias), axis=-1))
        inds = order[:, : self.top_k]
        cutoff = mx.take_along_axis(scores + self.bias, order[:, self.top_k : self.top_k + 1], axis=-1)
        picked = mx.take_along_axis(scores, inds, axis=-1)
        weights = picked / picked.sum(axis=-1, keepdims=True)
        return inds, weights, scores, mx.stop_gradient(cutoff[:, 0])

    def __call__(self, x, collect=None):
        shape = x.shape
        x = x.reshape(-1, shape[-1])
        inds, weights, scores, cutoff = self.route(x)
        n_experts = scores.shape[-1]

        routed = self.experts(self.down(x))
        mix = ((inds[..., None] == mx.arange(n_experts)) * weights[..., None]).sum(axis=1)
        u = (mix[..., None] * routed).sum(axis=1)

        out = self.up(self.u_norm(u))
        for expert in self.shared:
            out = out + expert(x)
        if collect is not None:
            collect.append(balanced_bias(mx.stop_gradient(scores), cutoff, self.top_k))
        return out.reshape(shape)


def balanced_bias(scores, cutoff, top_k):
    """Quantile balancing: pick each expert's bias so it receives its target load.

    `cutoff[i]` is the score a token's (top_k+1)-th choice had under the current bias,
    i.e. what expert j must beat to enter token i's top-k. Over the batch, expert j
    should beat that cutoff for exactly q = m*k/n tokens, so its bias is minus the
    (q+1)-th largest margin `scores[:, j] - cutoff`. The result is centred because a
    common offset does not change which experts are selected.
    """
    m, n = scores.shape
    target = int(round(m * top_k / n))
    margins = mx.sort(scores - cutoff[:, None], axis=0)
    index = min(max(m - target - 1, 0), m - 1)
    bias = -margins[index]
    return bias - bias.mean()
