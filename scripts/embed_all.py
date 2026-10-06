"""Grouped, resumable embedding of a window manifest through embed_windows.py.

Splits {tag}_windows.csv into contiguous groups of GROUP_SIZE rows, runs
embed_windows.py per group ({tag}_g{i}), then concatenates the pooled window
embeddings + meta in CSV order. Group intermediates (segment npz / chunk
emb npy) are deleted after the final concat, keeping disk use at ~1.4 GB
transient per group instead of ~45 GB for the full 240-patient cohort.

Resumable: groups whose final outputs exist are skipped, so a killed run
resumes where it stopped. Run dev and test tags as two parallel OS processes.

Usage: python scripts/embed_all.py --tag dev [--group-size 2000] [--keep-intermediates]
"""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "scripts"))

from config import EMBED_DIR, GROUP_SIZE  # noqa: E402


def group_paths(tag, i):
    d = EMBED_DIR
    return (d / f"{tag}_g{i}_window_ecgfm.npy", d / f"{tag}_g{i}_window_papagei.npy",
            d / f"{tag}_g{i}_meta.csv")


def intermediates(tag, i):
    d = EMBED_DIR
    return [d / f"{tag}_g{i}_ecg_seg.npz", d / f"{tag}_g{i}_ecg_emb.npy",
            d / f"{tag}_g{i}_ppg_seg.npz", d / f"{tag}_g{i}_ppg_emb.npy"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, choices=("dev", "test"))
    ap.add_argument("--group-size", type=int, default=GROUP_SIZE)
    ap.add_argument("--keep-intermediates", action="store_true")
    args = ap.parse_args()

    windows_csv = EMBED_DIR / f"{args.tag}_windows.csv"
    df = pd.read_csv(windows_csv)
    n = len(df)
    final = [EMBED_DIR / f"{args.tag}_window_ecgfm.npy",
             EMBED_DIR / f"{args.tag}_window_papagei.npy",
             EMBED_DIR / f"{args.tag}_meta.csv"]
    if all(p.exists() for p in final):
        print(f"[{args.tag}] final embeddings already cached, skipping")
        return
    bounds = list(range(0, n, args.group_size)) + [n]
    n_groups = len(bounds) - 1
    print(f"[{args.tag}] {n} windows -> {n_groups} groups of "
          f"{args.group_size}")

    for i in range(n_groups):
        lo, hi = bounds[i], bounds[i + 1]
        outs = group_paths(args.tag, i)
        if all(p.exists() for p in outs):
            print(f"[{args.tag}] group {i} already embedded, skipping")
            continue
        group_csv = EMBED_DIR / f"{args.tag}_g{i}_windows.csv"
        df.iloc[lo:hi].to_csv(group_csv, index=False)
        res = subprocess.run(
            [sys.executable, str(PROJECT / "scripts/embed_windows.py"),
             "--windows", str(group_csv), "--tag", f"{args.tag}_g{i}"],
            capture_output=True, text=True, timeout=7200)
        if res.returncode != 0:
            print(res.stdout)
            print(res.stderr, file=sys.stderr)
            raise RuntimeError(f"group {i} embedding failed")
        tail = (res.stdout or res.stderr).strip().splitlines()[-2:]
        print(f"[{args.tag}] group {i}: {' | '.join(tail)}")
        if not all(p.exists() for p in outs):
            raise RuntimeError(f"group {i} outputs missing after run")
        # free transient disk immediately (segments can be regenerated
        # deterministically via synth_signals.cut_segment)
        for p in intermediates(args.tag, i):
            p.unlink(missing_ok=True)
        group_csv.unlink(missing_ok=True)

    # ---------- final ordered concat -------------------------------------
    ecg_parts, ppg_parts, meta_parts = [], [], []
    for i in range(n_groups):
        e, p, m = group_paths(args.tag, i)
        ecg_parts.append(np.load(e))
        ppg_parts.append(np.load(p))
        gmeta = pd.read_csv(m)
        assert len(gmeta) == len(ecg_parts[-1]) == len(ppg_parts[-1]), \
            f"group {i} row counts out of sync"
        meta_parts.append(gmeta)
    ecg = np.concatenate(ecg_parts)
    ppg = np.concatenate(ppg_parts)
    meta = pd.concat(meta_parts, ignore_index=True)
    assert len(ecg) == len(ppg) == len(meta) == n, \
        f"concat row count {len(ecg)} != manifest rows {n}"
    np.save(final[0], ecg)
    np.save(final[1], ppg)
    meta.to_csv(final[2], index=False)
    print(f"[{args.tag}] concat done: {n} windows -> {args.tag}_window_*.npy + "
          f"{args.tag}_meta.csv")

    if not args.keep_intermediates:
        for i in range(n_groups):
            for p in list(group_paths(args.tag, i)):
                p.unlink(missing_ok=True)
        print(f"[{args.tag}] group files cleaned")


if __name__ == "__main__":
    main()