import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from delta_lm import Config, DeltaLM
from delta_lm.attnres import DepthMixer, run_with_attnres
from delta_lm.layers import KimiDeltaAttention, LatentMoE, SiTUGLU, balanced_bias
from delta_lm.tasks import AssociativeRecall


def small(**kw):
    base = dict(vocab=32, dim=32, heads=2, head_dim=16, n_routed=8, top_k=2, latent=16,
                routed_hidden=16, shared_hidden=32, mla_latent=16, repeats=1)
    base.update(kw)
    return Config(**base)


def test_situ_glu_is_bounded_even_for_huge_inputs():
    layer = SiTUGLU(8, 8)
    x = mx.random.normal((4, 8)) * 1e4
    assert float(mx.abs(layer.hidden(x)).max()) <= layer.bound_gate * layer.bound_up + 1e-3


def test_situ_glu_tracks_swiglu_near_zero():
    layer = SiTUGLU(8, 8)
    x = mx.random.normal((4, 8)) * 0.05
    g, u = layer.gate_proj(x), layer.up_proj(x)
    swiglu = g * mx.sigmoid(g) * u
    np.testing.assert_allclose(np.array(layer.hidden(x)), np.array(swiglu), atol=1e-4)


def test_delta_attention_chunk_and_recurrent_layers_agree():
    mx.random.seed(1)
    chunked = KimiDeltaAttention(32, 2, 16, impl="chunk")
    stepped = KimiDeltaAttention(32, 2, 16, impl="recurrent")
    stepped.update(chunked.parameters())
    x = mx.random.normal((2, 37, 32))
    np.testing.assert_allclose(np.array(chunked(x)), np.array(stepped(x)), atol=2e-4, rtol=2e-4)


def test_quantile_balancing_flattens_a_skewed_router():
    mx.random.seed(0)
    m, n, k = 512, 8, 2
    scores = mx.sigmoid(mx.random.normal((m, n)) + mx.linspace(-2, 2, n))
    bias = mx.zeros((n,))

    def loads(bias):
        order = mx.argsort(-(scores + bias), axis=-1)[:, :k]
        return np.bincount(np.array(order).ravel(), minlength=n)

    before = loads(bias)
    for _ in range(5):
        biased = scores + bias
        order = mx.argsort(-biased, axis=-1)
        cutoff = mx.take_along_axis(biased, order[:, k : k + 1], axis=-1)[:, 0]
        bias = balanced_bias(scores, cutoff, k)
    after = loads(bias)
    target = m * k / n
    assert before.max() > 1.8 * target
    assert after.max() < 1.15 * target and after.min() > 0.85 * target


def test_latent_moe_routing_weights_and_frozen_bias():
    moe = LatentMoE(32, 16, 8, 2, 16, 2, 32)
    x = mx.random.normal((10, 32))
    inds, weights, scores, _ = moe.route(x)
    assert inds.shape == (10, 2)
    np.testing.assert_allclose(np.array(weights.sum(-1)), 1.0, atol=1e-5)
    names = [n for n, _ in tree_flatten(moe.trainable_parameters())]
    assert "bias" not in names and "bias" in [n for n, _ in tree_flatten(moe.parameters())]


def test_bias_steers_selection_but_not_mixing_weights():
    moe = LatentMoE(32, 16, 8, 2, 16, 2, 32)
    x = mx.random.normal((6, 32))
    _, w0, scores, _ = moe.route(x)
    moe.bias = mx.array([0.0] * 7 + [5.0])
    inds, w1, _, _ = moe.route(x)
    assert (np.array(inds) == 7).any(axis=1).all()
    picked = np.take_along_axis(np.array(scores), np.array(inds), axis=1)
    np.testing.assert_allclose(np.array(w1), picked / picked.sum(1, keepdims=True), atol=1e-5)


def test_attnres_uniform_average_at_init_and_block_sources():
    mixer = DepthMixer(4, n_layers=6)
    seen = []

    class Probe:
        def __init__(self, tag):
            self.tag = tag

        def __call__(self, h):
            seen.append(h)
            return mx.full(h.shape, float(self.tag))

    emb = mx.zeros((1, 2, 4))
    out = run_with_attnres(mixer, [Probe(i + 1) for i in range(6)], emb, 3)
    # layer 0 sees only the embedding; layer 1 averages embedding and layer 0's partial sum
    assert float(seen[0].mean()) == 0.0
    assert float(seen[1].mean()) == pytest.approx(0.5)
    # layer 3 is the first of block 2: sources are the embedding and block 1's sum (1+2+3)
    assert float(seen[3].mean()) == pytest.approx(3.0)
    assert out.shape == emb.shape


@pytest.mark.parametrize("attn_res", [True, False])
@pytest.mark.parametrize("use_moe", [True, False])
def test_model_is_causal(attn_res, use_moe):
    mx.random.seed(2)
    model = DeltaLM(small(attn_res=attn_res, use_moe=use_moe))
    a = mx.random.randint(0, 32, (2, 40))
    b = mx.concatenate([a[:, :25], mx.random.randint(0, 32, (2, 15))], axis=1)
    la, lb = np.array(model(a)), np.array(model(b))
    np.testing.assert_allclose(la[:, :25], lb[:, :25], atol=1e-4)
    assert not np.allclose(la[:, 25:], lb[:, 25:], atol=1e-4)


def test_layer_layout_matches_pattern_and_ends_global():
    model = DeltaLM(small(pattern="KKKA", repeats=2))
    assert model.kinds == list("KKKAKKKA") + ["A"]
    assert len(model.layers) == 2 * len(model.kinds)  # each mixer is followed by an FFN
    assert len(model.moes()) == len(model.kinds)


def test_moe_layers_report_updated_bias():
    model = DeltaLM(small())
    collect = []
    model(mx.random.randint(0, 32, (2, 32)), collect)
    assert len(collect) == len(model.moes())
    model.apply_bias_updates(collect)
    assert abs(float(model.moes()[0].bias.mean())) < 1e-5


def test_recall_task_is_answerable_and_masked():
    task = AssociativeRecall(n_pairs=6, seed=3)
    seq, mask = task.sample(5)
    seq, mask = np.array(seq), np.array(mask)
    assert seq.shape == (5, 24) and mask.shape == (5, 23)
    for row, m in zip(seq, mask):
        table = dict(zip(row[0:12:2], row[1:12:2]))
        for pos in np.nonzero(m)[0]:
            assert row[pos + 1] == table[row[pos]]
    assert mask.sum() == 5 * 6
