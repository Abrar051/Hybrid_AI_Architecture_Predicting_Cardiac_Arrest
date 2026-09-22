"""Extract ECG-FM embeddings from cached 12-lead 500 Hz segments.

Run with the ecgfm_env interpreter (fairseq-signals, torch 1.13):
    ~/anaconda3/envs/ecgfm_env/bin/python scripts/extract_ecgfm.py \
        --input cache/embeddings/demo_ecg.npz --output cache/embeddings/demo_ecgfm.npy \
        --checkpoint weights/mimic_iv_ecg_physionet_pretrained.pt

Input npz: 'ecg' float32 array (n_segments, 12, 2500), z-scored per segment.
Output:   (n_segments, 768) float32 embeddings, mean-pooled over time.
"""
import argparse

import numpy as np
import torch

from fairseq_signals.models import build_model_from_checkpoint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    x = np.load(args.input)["ecg"].astype(np.float32)
    assert x.ndim == 3 and x.shape[1] == 12, f"expected (n, 12, 2500), got {x.shape}"

    model = build_model_from_checkpoint(checkpoint_path=args.checkpoint)
    model.eval()

    embs = []
    with torch.no_grad():
        for i in range(0, len(x), args.batch_size):
            src = torch.from_numpy(x[i:i + args.batch_size])
            # features_only path: clean encoder output, no pretraining head
            res = model.extract_features(src, None)      # mask=False
            h = res["x"]                                 # (B, T, 768) batch-first
            pad = res.get("padding_mask")                # (B, T)
            if pad is not None:
                h = h.masked_fill(pad.unsqueeze(-1), 0.0)
                h = h.sum(dim=1) / (~pad).sum(dim=1, keepdim=True).clamp(min=1)
            else:
                h = h.mean(dim=1)                        # (B, 768)
            embs.append(h.numpy())
    embs = np.concatenate(embs, axis=0).astype(np.float32)
    np.save(args.output, embs)
    print(f"ECG-FM: {len(x)} segments -> {embs.shape} saved to {args.output}")


if __name__ == "__main__":
    main()