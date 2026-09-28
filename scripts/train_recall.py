"""Compare layer mixes on multi-query associative recall.

    python scripts/train_recall.py
    python scripts/train_recall.py --steps 4000 --pairs 8
"""

import argparse
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from delta_lm import Config, DeltaLM  # noqa: E402
from delta_lm.tasks import AssociativeRecall  # noqa: E402

VARIANTS = {
    "hybrid KKKA + final A": dict(pattern="KKKA", repeats=1, final_global=True),
    "delta only KKKK": dict(pattern="KKKK", repeats=1, final_global=False),
    "global only AAAA (NoPE)": dict(pattern="AAAA", repeats=1, final_global=False),
}


def run(name, overrides, task, steps, batch, lr, seed):
    mx.random.seed(seed)
    cfg = Config(vocab=task.vocab, **overrides)
    model = DeltaLM(cfg)
    mx.eval(model.parameters())
    params = sum(v.size for _, v in tree_flatten(model.parameters()))
    opt = optim.AdamW(learning_rate=optim.cosine_decay(lr, steps), weight_decay=0.01)

    def loss_fn(model, seq, mask):
        collect = []
        logits = model(seq[:, :-1], collect)
        ce = nn.losses.cross_entropy(logits, seq[:, 1:])
        loss = (ce * mask).sum() / mask.sum()
        hits = ((logits.argmax(-1) == seq[:, 1:]) * mask).sum() / mask.sum()
        return loss, (hits, mx.stack(collect) if collect else mx.zeros((0,)))

    step = nn.value_and_grad(model, loss_fn)
    start, acc = time.perf_counter(), 0.0
    for i in range(1, steps + 1):
        seq, mask = task.sample(batch)
        (loss, (hits, biases)), grads = step(model, seq, mask)
        grads, _ = optim.clip_grad_norm(grads, 1.0)
        opt.update(model, grads)
        model.apply_bias_updates(list(biases))
        mx.eval(model.parameters(), opt.state, loss, hits)
        acc = 0.95 * acc + 0.05 * float(hits) if i > 1 else float(hits)
        if i % 500 == 0:
            print(f"  [{name}] step {i:5d}  loss {float(loss):.3f}  acc~{acc:.3f}")
    seconds = time.perf_counter() - start

    test = AssociativeRecall(task.n_keys, task.n_values, task.n_pairs, seed=999)
    correct = 0.0
    for _ in range(10):
        seq, mask = test.sample(256)
        pred = model(seq[:, :-1]).argmax(-1)
        correct += float(((pred == seq[:, 1:]) * mask).sum() / mask.sum())
    return params, correct / 10, seconds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--pairs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    task = AssociativeRecall(n_pairs=args.pairs, seed=args.seed)
    print(f"chance accuracy is about {1 / task.n_values:.3f}\n")
    rows = []
    for name, overrides in VARIANTS.items():
        params, acc, seconds = run(name, overrides, task, args.steps, args.batch, args.lr, args.seed)
        rows.append((name, params, acc, seconds))

    print("\n| variant | params | recall accuracy | train time |")
    print("|---|---:|---:|---:|")
    for name, params, acc, seconds in rows:
        print(f"| {name} | {params:,} | {acc:.3f} | {seconds:.0f}s |")


if __name__ == "__main__":
    main()
