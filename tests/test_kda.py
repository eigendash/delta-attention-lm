import mlx.core as mx
import numpy as np
import pytest

from delta_lm.kda import chunk_kda, recurrent_kda


def inputs(B=2, H=2, T=40, dk=8, dv=6, g_min=-5.0, seed=0):
    mx.random.seed(seed)
    q = mx.random.normal((B, H, T, dk))
    k = mx.random.normal((B, H, T, dk))
    q = q / mx.linalg.norm(q, axis=-1, keepdims=True)
    k = k / mx.linalg.norm(k, axis=-1, keepdims=True)
    v = mx.random.normal((B, H, T, dv))
    log_decay = g_min * mx.sigmoid(mx.random.normal((B, H, T, dk)))
    beta = mx.sigmoid(mx.random.normal((B, H, T)))
    return q, k, v, log_decay, beta


@pytest.mark.parametrize("T", [16, 40, 7, 1])
@pytest.mark.parametrize("chunk", [4, 16])
def test_chunk_matches_recurrent(T, chunk):
    args = inputs(T=T)
    ref, ref_state = recurrent_kda(*args)
    out, state = chunk_kda(*args, chunk=chunk)
    np.testing.assert_allclose(np.array(out), np.array(ref), atol=2e-4, rtol=2e-4)
    np.testing.assert_allclose(np.array(state), np.array(ref_state), atol=2e-4, rtol=2e-4)


def test_state_carries_across_calls():
    args = inputs(T=48)
    full, full_state = chunk_kda(*args)
    first = [a[:, :, :32] for a in args]
    second = [a[:, :, 32:] for a in args]
    out1, mid = chunk_kda(*first)
    out2, end = chunk_kda(*second, state=mid)
    np.testing.assert_allclose(
        np.array(mx.concatenate([out1, out2], axis=2)), np.array(full), atol=2e-4, rtol=2e-4
    )
    np.testing.assert_allclose(np.array(end), np.array(full_state), atol=2e-4, rtol=2e-4)


def test_causal_future_tokens_do_not_leak():
    args = list(inputs(T=32))
    base, _ = chunk_kda(*args)
    changed = [a for a in args]
    changed[2] = mx.concatenate([args[2][:, :, :20], mx.ones_like(args[2][:, :, 20:])], axis=2)
    out, _ = chunk_kda(*changed)
    np.testing.assert_allclose(np.array(out[:, :, :20]), np.array(base[:, :, :20]), atol=1e-6)


def test_gradients_are_finite_at_the_decay_bound():
    q, k, v, log_decay, beta = inputs(T=32, g_min=-5.0)
    log_decay = mx.full(log_decay.shape, -5.0)

    def loss(q, k, v, log_decay, beta):
        return chunk_kda(q, k, v, log_decay, beta)[0].sum()

    grads = mx.grad(loss, argnums=(0, 1, 2, 3, 4))(q, k, v, log_decay, beta)
    for g in grads:
        assert np.isfinite(np.array(g)).all()


def test_delta_rule_overwrites_a_repeated_key():
    # With beta=1, unit key and no decay, writing (k, v2) after (k, v1) replaces v1.
    k = mx.array([[[[1.0, 0.0], [1.0, 0.0]]]])
    v = mx.array([[[[1.0, 0.0], [0.0, 1.0]]]])
    q = k
    out, state = recurrent_kda(q, k, v, mx.zeros_like(k), mx.ones((1, 1, 2)))
    np.testing.assert_allclose(np.array(out)[0, 0, 1], [0.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(np.array(state)[0, 0, 0], [0.0, 1.0], atol=1e-6)
