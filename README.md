# delta-attention-lm

A small language model that mixes linear-time delta-rule attention with a few global attention layers, a latent mixture-of-experts feed-forward, and attention over depth instead of a plain residual stream. It runs on Apple silicon through MLX.

The architecture follows section 2 of *Kimi K3: Open Frontier Intelligence* (Kimi Team, arXiv:2607.24653). That model has 2.8T parameters. This one has about 500k, so it is a study of how the pieces behave and fit together, not a reproduction of anything in the paper's results. The code was written from the paper's equations and description.

## What is in here

**Delta attention** (`delta_lm/kda.py`, `delta_lm/layers.py`). Each head keeps a state matrix `S` that is decayed per key channel and then corrected with the delta rule, so writing a key again overwrites the old value instead of piling onto it. The recurrent form is the reference. The chunkwise form splits the sequence into chunks of 16, solves a small triangular system inside each chunk, and carries `S` across chunks. I derived the chunk equations from the recurrence and test them against it. The inverse of the unit lower-triangular matrix is computed as a product of `(I + M^(2^j))` factors, which only uses matmuls and keeps gradients working in MLX.

The decay is bounded below: the per-step log decay is `g_min * sigmoid(exp(A) * z)` with `g_min = -5`, so no retention factor falls under `e^-5`. Inside a chunk of 16 this keeps the reciprocal cumulative decay under `e^80`, which fits in float32. The output goes through a per-head RMSNorm and a full-rank sigmoid gate.

**Global attention** is multi-head latent attention with no positional encoding and the same style of output gate. The delta layers carry the position information through their short causal convolutions and decay.

**Latent MoE** (`LatentMoE`). Two shared full-width experts run on every token. The routed experts work in a narrower latent space: `W_down` maps to the latent width, the selected experts run there, their weighted sum goes through an RMSNorm, and `W_up` maps back. Expert activations use a GLU whose gate and up branches are both soft-capped with `b*tanh(x/b)`, so the hidden activation is bounded by `b1*b2`. Near zero it matches SwiGLU.

Routing uses sigmoid scores and a per-expert bias that affects which experts are picked but not how their outputs are weighted. The bias is refreshed by quantile balancing: after each forward pass, each expert's bias is set so it would receive its target share of tokens, using the score each token's `(k+1)`-th choice had to beat. I use the exact quantile over the batch. The paper estimates it from histograms because its batches are spread across many machines.

**Attention residuals** (`delta_lm/attnres.py`). Each sublayer has a learned query. Its input is a softmax-weighted mix of the token embedding, the summed outputs of finished blocks, and the running sum of the current block, scored against RMS-normalised keys. The queries start at zero, so the mix starts as a plain average.

The stack follows a repeating pattern of delta layers and global layers, each followed by an MoE, and ends with one extra global layer. The pattern is configurable (`K` for delta, `A` for global).

## A small experiment

`scripts/train_recall.py` trains three layer mixes on multi-query associative recall: eight key-value pairs, then the keys again in shuffled order, and the model has to produce each value. Chance is about 3%. Each model trains for 3000 steps with AdamW and a cosine schedule. Single seed.

| layers | parameters | recall accuracy |
|---|---:|---:|
| 3 delta + 1 global, plus a final global (default) | 511,852 | 0.996 |
| 4 delta, no global | 419,472 | 0.995 |
| 4 global, no positional encoding | 398,656 | 0.282 |

Two things to note. First, the hybrid and the delta-only model are indistinguishable on this task, so it says nothing about why the hybrid helps at scale. It does show that the delta layers carry recall by themselves. Second, global attention with no positional encoding does poorly here, which is consistent with the paper's reason for pairing those layers with delta layers that know about position. A causal model without positions can only guess where the value after a key sits.

The first version of this experiment used a learning rate of 3e-3 and the hybrid stayed at 8.7%, while the delta-only model reached 99%. Dropping to 1e-3 fixed it. I also tried zero-initialising the global layer's output projection and that did not help, so the problem was step size, not the initial scale of the global layers. The numbers above use 1e-3 for every variant.

## Running it

```
pip install -e ".[dev]"
pytest
python scripts/train_recall.py
```

The tests check the chunkwise form against the recurrent one for several lengths and chunk sizes, state carried across calls, causality of both the layers and the full model, gradient finiteness at the decay bound, that quantile balancing flattens a skewed router, and the attention-residual source bookkeeping.

## Limits

This is the single-device, small-scale version. There is no context parallelism, no kernel fusion, and no custom Metal kernel: the chunk loop runs as ordinary MLX operations. Experts are evaluated densely over all tokens and masked, which is simple and fine at 16 experts but would not scale. There is no vision tower, no Muon optimizer, and no long-context training. The decay-bias initialisation is my own choice. The paper cites earlier work for it, and I did not try to match it.
