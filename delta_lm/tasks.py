"""Multi-query associative recall.

The sequence lists n key/value pairs, then lists the same keys again in a shuffled
order, each followed by its value. The model is scored only on predicting those
second-pass values, which it can only do by remembering the first pass.

    k3 v3 k1 v1 k2 v2 | k2 ? k3 ? k1 ?      (the ? are the scored positions)
"""

import mlx.core as mx
import numpy as np


class AssociativeRecall:
    def __init__(self, n_keys=32, n_values=32, n_pairs=8, seed=0):
        self.n_keys, self.n_values, self.n_pairs = n_keys, n_values, n_pairs
        self.vocab = n_keys + n_values
        self.length = 4 * n_pairs
        self.rng = np.random.default_rng(seed)

    def sample(self, batch):
        n, rng = self.n_pairs, self.rng
        keys = np.argsort(rng.random((batch, self.n_keys)), axis=1)[:, :n]
        values = rng.integers(self.n_keys, self.vocab, (batch, n))
        order = np.argsort(rng.random((batch, n)), axis=1)
        rows = np.arange(batch)[:, None]

        seq = np.zeros((batch, self.length), dtype=np.int32)
        seq[:, 0 : 2 * n : 2], seq[:, 1 : 2 * n : 2] = keys, values
        seq[:, 2 * n :: 2], seq[:, 2 * n + 1 :: 2] = keys[rows, order], values[rows, order]

        mask = np.zeros((batch, self.length - 1), dtype=np.float32)
        mask[:, 2 * n :: 2] = 1.0  # targets that are second-pass values
        return mx.array(seq), mx.array(mask)
