"""Extract PaPaGei-S PPG embeddings from cached 125 Hz segments.

Run with the papagei_env interpreter:
    ~/anaconda3/envs/papagei_env/bin/python scripts/extract_papagei.py \
        --input cache/embeddings/demo_ppg.npz --output cache/embeddings/demo_papagei.npy \
        --weights weights/papagei_s.pt

Input npz: 'ppg' float32 array (n_segments, 1250) at 125 Hz, z-scored per segment.
Output:   (n_segments, 512) float32 embeddings (first element of the model's
          output tuple, matching papagei's own feature_extraction_papagei.py).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent / "third_party" / "papagei-foundation-model"
sys.path.insert(0, str(REPO))

from models.resnet import ResNet1DMoE  # noqa: E402


def load_model_without_module_prefix(model, checkpoint_path):
    """Same logic as papagei's linearprobing.utils.load_model_without_module_prefix."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    new_state_dict = {k[7:] if k.startswith("module.") else k: v
                      for k, v in checkpoint.items()}
    model.load_state_dict(new_state_dict)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    x = np.load(args.input)["ppg"].astype(np.float32)
    assert x.ndim == 2 and x.shape[1] == 1250, f"expected (n, 1250), got {x.shape}"

    model = ResNet1DMoE(in_channels=1, base_filters=32, kernel_size=3, stride=2,
                        groups=1, n_block=18, n_classes=512, n_experts=3)
    load_model_without_module_prefix(model, args.weights)
    model.eval()

    embs = []
    with torch.inference_mode():
        for i in range(0, len(x), args.batch_size):
            src = torch.from_numpy(x[i:i + args.batch_size]).unsqueeze(1)
            outputs = model(src)
            if isinstance(outputs, tuple):
                outputs = outputs[0]          # (embeddings, experts, gating)
            embs.append(outputs.cpu().numpy())
    embs = np.concatenate(embs, axis=0).astype(np.float32)
    np.save(args.output, embs)
    print(f"PaPaGei-S: {len(x)} segments -> {embs.shape} saved to {args.output}")


if __name__ == "__main__":
    main()