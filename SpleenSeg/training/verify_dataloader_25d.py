from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

from SpleenSeg.preprocessing.transforms import PreprocessConfig
from SpleenSeg.training.dataset_25d import DecathlonSpleen25DDataset, Slice25DConfig


def _save_overlay(image_chw: np.ndarray, label_1hw: np.ndarray, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Visualize the center channel
    c = image_chw.shape[0] // 2
    img = image_chw[c]
    m = label_1hw[0]

    vmin = float(np.percentile(img, 1.0))
    vmax = float(np.percentile(img, 99.0))

    fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=150)
    ax.imshow(img.T, cmap="gray", vmin=vmin, vmax=vmax, origin="lower", interpolation="bilinear")
    ax.imshow(np.ma.masked_where(m.T <= 0, m.T), cmap="Reds", alpha=0.40, origin="lower", interpolation="nearest")
    ax.set_title("2.5D sample (center slice)")
    ax.axis("off")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 3: Verify 2.5D dataloader.")
    parser.add_argument("--dataset-root", type=Path, default=Path("Task09_Spleen"))
    parser.add_argument("--num-slices", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-cases", type=int, default=1, help="Limit upfront preprocessing to N cases for quick verification")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("artifacts/step3_25d_sample.png"))
    args = parser.parse_args()

    pre_cfg = PreprocessConfig()
    slice_cfg = Slice25DConfig(num_slices=args.num_slices, positive_only=True, negative_ratio=1.0)

    ds = DecathlonSpleen25DDataset(
        dataset_root=args.dataset_root,
        preprocess_config=pre_cfg,
        slice_config=slice_cfg,
        augment=False,
        max_cases=int(args.max_cases) if args.max_cases and args.max_cases > 0 else None,
        verbose=bool(args.verbose),
        seed=0,
    )

    print(f"Dataset size (slices): {len(ds)}")
    sample = ds[0]
    print(f"Single sample image shape: {tuple(sample['image'].shape)} dtype={sample['image'].dtype}")
    print(f"Single sample label shape: {tuple(sample['label'].shape)} dtype={sample['label'].dtype}")
    print(f"Meta: {sample['meta']}")

    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    batch = next(iter(dl))
    print(f"Batch image shape: {tuple(batch['image'].shape)}")
    print(f"Batch label shape: {tuple(batch['label'].shape)}")

    img0 = batch["image"][0].numpy()
    lab0 = batch["label"][0].numpy()
    _save_overlay(img0, lab0, args.output)
    print(f"Saved overlay: {args.output.resolve()}")


if __name__ == "__main__":
    main()
