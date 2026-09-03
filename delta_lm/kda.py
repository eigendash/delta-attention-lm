"""Gated delta-rule linear attention with channel-wise decay.

Per head, with state S in R^{dk x dv}:

    S_t = (I - b_t k_t k_t^T) Diag(a_t) S_{t-1} + b_t k_t v_t^T
    o_t = S_t^T q_t

`recurrent_kda` evaluates this one token at a time and is the reference.
`chunk_kda` evaluates it chunk by chunk: parallel inside a chunk, recurrent across
chunks. Inside a chunk, with g the cumulative log decay and gamma = exp(g):

    u_t = b_t (v_t - S_0^T (gamma_t * k_t)) - b_t sum_{r<t} A_tr u_r
    A_tr = (gamma_t * k_t) . (k_r / gamma_r)

which is a unit lower-triangular solve. We invert that matrix with a short product
of (I + M^(2^j)) factors, because M is strictly lower triangular and so nilpotent.
That keeps everything as plain matmuls with working gradients.

Shapes: q, k (B, H, T, dk); v (B, H, T, dv); log_decay (B, H, T, dk) and negative;
beta (B, H, T).
"""

import mlx.core as mx


def recurrent_kda(q, k, v, log_decay, beta, state=None):
    B, H, T, dk = q.shape
    dv = v.shape[-1]
    S = mx.zeros((B, H, dk, dv), dtype=q.dtype) if state is None else state
    outs = []
    for t in range(T):
        kt = k[:, :, t]
        S = mx.exp(log_decay[:, :, t])[..., None] * S
        pred = (kt[..., None] * S).sum(axis=-2)
        S = S + beta[:, :, t, None, None] * kt[..., None] * (v[:, :, t] - pred)[..., None, :]
        outs.append((q[:, :, t, :, None] * S).sum(axis=-2))
    return mx.stack(outs, axis=2), S


def _inverse_unit_lower(M, size):
    """(I + M)^-1 for strictly lower-triangular M, via (I - M)(I + M^2)(I + M^4)..."""
    eye = mx.eye(size, dtype=M.dtype)
    inv = eye - M
    power, span = M @ M, 2
    while span < size:
        inv = inv @ (eye + power)
        power, span = power @ power, span * 2
    return inv


def chunk_kda(q, k, v, log_decay, beta, state=None, chunk=16):
    if chunk & (chunk - 1):
        raise ValueError("chunk must be a power of two")
    B, H, T, dk = q.shape
    dv = v.shape[-1]
    pad = (-T) % chunk
    if pad:
        # zero beta writes nothing and zero log-decay keeps the state, so padding is inert
        p3 = [(0, 0), (0, 0), (0, pad), (0, 0)]
        q, k, v, log_decay = (mx.pad(x, p3) for x in (q, k, v, log_decay))
        beta = mx.pad(beta, [(0, 0), (0, 0), (0, pad)])
    N = (T + pad) // chunk

    def split(x):
        return x.reshape(B, H, N, chunk, *x.shape[3:])

    q, k, v, log_decay, beta = map(split, (q, k, v, log_decay, beta))

    g = mx.cumsum(log_decay, axis=3)
    gamma, inv_gamma = mx.exp(g), mx.exp(-g)
    q_g, k_g, k_inv = q * gamma, k * gamma, k * inv_gamma

    attn = mx.tril(q_g @ k_inv.swapaxes(-1, -2))
    M = beta[..., None] * mx.tril(k_g @ k_inv.swapaxes(-1, -2), k=-1)
    T_mat = _inverse_unit_lower(M, chunk)
    U = T_mat @ (beta[..., None] * v)
    W = T_mat @ (beta[..., None] * k_g)

    last = g[:, :, :, -1:, :]
    k_tail = k * mx.exp(last - g)
    gamma_last = mx.exp(last[:, :, :, 0, :])

    S = mx.zeros((B, H, dk, dv), dtype=q.dtype) if state is None else state
    outs = []
    for n in range(N):
        v_new = U[:, :, n] - W[:, :, n] @ S
        outs.append(q_g[:, :, n] @ S + attn[:, :, n] @ v_new)
        S = gamma_last[:, :, n, :, None] * S + k_tail[:, :, n].swapaxes(-1, -2) @ v_new
    out = mx.stack(outs, axis=2).reshape(B, H, N * chunk, dv)
    return out[:, :, :T], S
