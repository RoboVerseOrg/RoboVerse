"""Train the compact chunked-BC LIBERO policy on demo HDF5s (GPU env).

Run in a torch-2.7/cu128 env (sm_120-capable), e.g. ``roboverse``::

    python -m scripts.policy.train_bc_libero \\
        --demos third_party/libero_datasets/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5 \\
        --epochs 80 --out scripts/policy/ckpt/bc_alphabet_soup.pt

Saves a checkpoint (state_dict + proprio normalization + meta) that the CPU
closed-loop evaluator loads verbatim in the ``liberoplus`` env.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import h5py
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bc_model import _MEAN, _STD, ACTION_DIM, CHUNK, PROPRIO_KEYS, ChunkedBCPolicy


def _load_samples(demo_files: list[str], chunk: int):
    agent, hand, proprio, actions, mask = [], [], [], [], []
    for f in demo_files:
        with h5py.File(f, "r") as h:
            data = h["data"]
            for dk in data.keys():
                g = data[dk]
                T = g["actions"].shape[0]
                av = np.asarray(g["obs"]["agentview_rgb"])
                eh = np.asarray(g["obs"]["eye_in_hand_rgb"])
                pr = np.concatenate([np.asarray(g["obs"][k]) for k in PROPRIO_KEYS], axis=1).astype(np.float32)
                ac = np.asarray(g["actions"]).astype(np.float32)
                for t in range(T):
                    a = ac[t : t + chunk]
                    m = np.ones((chunk,), np.float32)
                    if a.shape[0] < chunk:
                        pad = chunk - a.shape[0]
                        m[a.shape[0] :] = 0.0
                        a = np.concatenate([a, np.zeros((pad, ACTION_DIM), np.float32)], axis=0)
                    agent.append(av[t])
                    hand.append(eh[t])
                    proprio.append(pr[t])
                    actions.append(a)
                    mask.append(m)
    return (
        np.asarray(agent),
        np.asarray(hand),
        np.asarray(proprio, np.float32),
        np.asarray(actions, np.float32),
        np.asarray(mask, np.float32),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", nargs="+", required=True, help="demo hdf5 path(s) or glob(s)")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--chunk", type=int, default=CHUNK)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    files: list[str] = []
    for d in args.demos:
        files.extend(sorted(glob.glob(d)) or [d])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} | demo files={len(files)}")

    av, eh, pr, ac, mk = _load_samples(files, args.chunk)
    print(f"samples={len(av)}  proprio_dim={pr.shape[1]}  chunk={args.chunk}")
    p_mean, p_std = pr.mean(0), pr.std(0) + 1e-6
    pr_n = (pr - p_mean) / p_std

    av_t = torch.from_numpy(av).float().div(255.0).permute(0, 3, 1, 2)
    eh_t = torch.from_numpy(eh).float().div(255.0).permute(0, 3, 1, 2)
    av_t = (av_t - _MEAN) / _STD
    eh_t = (eh_t - _MEAN) / _STD
    pr_t = torch.from_numpy(pr_n).float()
    ac_t = torch.from_numpy(ac).float()
    mk_t = torch.from_numpy(mk).float()

    ds = torch.utils.data.TensorDataset(av_t, eh_t, pr_t, ac_t, mk_t)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True)

    policy = ChunkedBCPolicy(chunk=args.chunk).to(device)
    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    l1 = nn.L1Loss(reduction="none")

    policy.train()
    for ep in range(args.epochs):
        tot, n = 0.0, 0
        for a_im, h_im, p, a, m in dl:
            a_im, h_im, p, a, m = (x.to(device, non_blocking=True) for x in (a_im, h_im, p, a, m))
            pred = policy(a_im, h_im, p)
            loss = (l1(pred, a).mean(-1) * m).sum() / m.sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * a.shape[0]
            n += a.shape[0]
        sched.step()
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"epoch {ep:3d}  L1={tot / n:.4f}  lr={sched.get_last_lr()[0]:.2e}", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(
        {
            "state_dict": policy.state_dict(),
            "proprio_mean": p_mean,
            "proprio_std": p_std,
            "chunk": args.chunk,
            "demo_files": files,
        },
        args.out,
    )
    print(f"saved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
