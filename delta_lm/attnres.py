"""Block attention residuals: softmax attention over depth instead of a plain residual sum.

Each layer owns a learned pseudo-query. Its input is a softmax-weighted mix of
earlier representations, scored against RMS-normalised keys so that a layer with a
large output cannot drown out the rest. To keep memory O(blocks) instead of
O(layers), layers are grouped into blocks and only these sources are kept:

    b_0              the token embedding
    b_1 .. b_{n-1}   sums of layer outputs from each finished block
    partial          the running sum inside the current block (from its 2nd layer on)
"""

import mlx.core as mx
import mlx.nn as nn


class DepthMixer(nn.Module):
    def __init__(self, dim, n_layers):
        super().__init__()
        # One query per layer plus one for the final read-out. Zeros start as a plain
        # average over sources.
        self.queries = mx.zeros((n_layers + 1, dim))

    def mix(self, index, sources):
        stack = mx.stack(sources, axis=0)
        keys = mx.fast.rms_norm(stack, mx.ones((stack.shape[-1],)), 1e-6)
        weights = mx.softmax((keys * self.queries[index]).sum(axis=-1), axis=0)
        return (weights[..., None] * stack).sum(axis=0)


def run_with_attnres(mixer, layers, embedding, block_size, collect=None):
    """Run `layers` (callables taking and returning (B, T, D)) with block AttnRes."""
    completed, partial = [embedding], None
    for i, layer in enumerate(layers):
        h = mixer.mix(i, completed + ([partial] if partial is not None else []))
        out = layer(h, collect) if collect is not None else layer(h)
        partial = out if partial is None else partial + out
        if (i + 1) % block_size == 0:
            completed.append(partial)
            partial = None
    return mixer.mix(len(layers), completed + ([partial] if partial is not None else []))
