from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from SpleenSeg.preprocessing.transforms import build_preprocessing_transforms, config_from_args


def _load_training_pairs(dataset_root: Path) -> list[tuple[Path, Path]]:
    dataset_json = dataset_root / "dataset.json"
    if not dataset_json.exists():
        raise FileNotFoundError(f"Missing dataset.json at: {dataset_json}")
    info = json.loads(dataset_json.read_text())

    items = info.get("training", [])
    if not items:
        raise ValueError("No 'training' items found in dataset.json")

    pairs: list[tuple[Path, Path]] = []
    for it in items:
        image_path = (dataset_root / it["image"]).resolve()
        label_path = (dataset_root / it["label"]).resolve()
        pairs.append((image_path, label_path))
    return pairs


def _case_id_from_image_path(image_path: Path) -> str:
    # Example filename: spleen_19.nii.gz
    name = image_path.name
    if name.endswith(".nii.gz"):
        return name[: -len(".nii.gz")]
    return image_path.stem


def _save_npz(out_dir: Path, case_id: str, image: np.ndarray, label: np.ndarray, spacing: tuple[float, float, float]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{case_id}.npz"
    np.savez_compressed(
        out_path,
        image=image.astype(np.float32, copy=False),
        label=label.astype(np.uint8, copy=False),
        spacing=np.asarray(spacing, dtype=np.float32),
    )
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess Task09_Spleen training set.")
    parser.add_argument("--dataset-root", type=Path, default=Path("Task09_Spleen"))
    parser.add_argument("--out-dir", type=Path, default=Path("data_processed/task09_spleen"))

    parser.add_argument("--hu-min", type=float, default=-200.0)
    parser.add_argument("--hu-max", type=float, default=300.0)
    parser.add_argument("--spacing", type=float, nargs=3, default=(1.5, 1.5, 1.5))
    parser.add_argument("--roi-size", type=int, nargs=3, default=(128, 128, 128))
    parser.add_argument("--axcodes", type=str, default="RAS")

    parser.add_argument("--limit", type=int, default=0, help="If >0, preprocess only first N cases")

    args = parser.parse_args()

    pairs = _load_training_pairs(args.dataset_root)
    if args.limit and args.limit > 0:
        pairs = pairs[: args.limit]

    cfg = config_from_args(
        hu_min=args.hu_min,
        hu_max=args.hu_max,
        target_spacing=args.spacing,
        roi_size=args.roi_size,
        axcodes=args.axcodes,
    )
    tfm = build_preprocessing_transforms(cfg)

    print("Preprocessing config:")
    print(f"  HU clip: [{cfg.hu_min}, {cfg.hu_max}]")
    print(f"  target spacing: {cfg.target_spacing}")
    print(f"  roi size: {cfg.roi_size}")
    print(f"  orientation: {cfg.axcodes}")
    print(f"  cases: {len(pairs)}")

    n_ok = 0
    for image_path, label_path in pairs:
        if not image_path.exists():
            raise FileNotFoundError(f"Missing image: {image_path}")
        if not label_path.exists():
            raise FileNotFoundError(f"Missing label: {label_path}")

        case_id = _case_id_from_image_path(image_path)
        sample = {"image": str(image_path), "label": str(label_path)}
        out = tfm(sample)

        # MONAI outputs channel-first arrays: [C, D, H, W]
        image = out["image"]
        label = out["label"]

        # Convert to numpy (works whether MONAI returned numpy or torch tensors)
        if hasattr(image, "detach"):
            image_np = image.detach().cpu().numpy()
        else:
            image_np = np.asarray(image)

        if hasattr(label, "detach"):
            label_np = label.detach().cpu().numpy()
        else:
            label_np = np.asarray(label)

        # Basic sanity checks
        if image_np.shape != (1, *cfg.roi_size):
            raise ValueError(f"Unexpected image shape {image_np.shape} for {case_id}")
        if label_np.shape != (1, *cfg.roi_size):
            raise ValueError(f"Unexpected label shape {label_np.shape} for {case_id}")

        # Ensure label is 0/1
        label_np = (label_np > 0).astype(np.uint8)

        out_path = _save_npz(
            args.out_dir,
            case_id,
            image=image_np,
            label=label_np,
            spacing=cfg.target_spacing,
        )

        n_ok += 1
        if n_ok <= 3:
            print(
                f"[{n_ok}/{len(pairs)}] {case_id}: saved {out_path} | "
                f"image min/max={float(image_np.min()):.3f}/{float(image_np.max()):.3f} | "
                f"label sum={int(label_np.sum())}"
            )

    print(f"Done. Preprocessed {n_ok} cases into: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
