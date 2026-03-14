import argparse
from pathlib import Path
import json

import nibabel as nib
import numpy as np


def get_case_paths(dataset_root: str | Path, split: str, index: int) -> tuple[Path, Path | None]:
    dataset_root = Path(dataset_root)
    dataset_json_path = dataset_root / "dataset.json"
    if not dataset_json_path.exists():
        raise FileNotFoundError(f"dataset.json not found at {dataset_json_path}")
    dataset_info = json.loads(dataset_json_path.read_text())

    if split == "train":
        items = dataset_info.get("training", [])
        if not items:
            raise ValueError("No 'training' items found in dataset.json")
        if index < 0 or index >= len(items):
            raise IndexError(f"Index {index} out of range for training set (n={len(items)})")
        image_rel = items[index]["image"]
        label_rel = items[index]["label"]
        image_path = (dataset_root / image_rel).resolve()
        label_path = (dataset_root / label_rel).resolve()
        return image_path, label_path

    if split == "test":
        items = dataset_info.get("test", [])
        if not items:
            raise ValueError("No 'test' items found in dataset.json")
        if index < 0 or index >= len(items):
            raise IndexError(f"Index {index} out of range for test set (n={len(items)})")
        image_rel = items[index]
        image_path = (dataset_root / image_rel).resolve()
        return image_path, None

    raise ValueError(f"Unknown split: {split}. Expected 'train' or 'test'.")

def describe_nifti(image: nib.Nifti1Image, name: str) -> None:
    print(f"{name}:")
    print(f"  shape: {image.shape}")
    print(f"  voxel_spacing (header zooms): {image.header.get_zooms()}")
    print(f"  dtype (on-disk): {image.get_data_dtype()}")

def render_middle_slice(ct_data: np.ndarray, mask_data: np.ndarray | None = None, output_path: Path | None = None, show: bool = False):
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z_mid = ct_data.shape[2] // 2
    ct_slice = ct_data[:, :, z_mid]
    vmin = float(np.percentile(ct_slice, 1.0))
    vmax = float(np.percentile(ct_slice, 99.0))

    plt.figure(figsize=(8, 8), dpi=150)
    plt.imshow(ct_slice.T, cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
    if mask_data is not None:
        mask_slice = mask_data[:, :, z_mid]
        plt.imshow(np.ma.masked_where(mask_slice.T <= 0, mask_slice.T), cmap="Reds", alpha=0.35, origin="lower")
    plt.axis("off")
    if show:
        plt.show()
        plt.close()
        return None
    else:
        if output_path is None:
            raise ValueError("output_path must be provided when show=False")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
        plt.close()
        return output_path



def main():
    parser = argparse.ArgumentParser(description="Verify the dataset")

    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("Task09_Spleen"),
        help="Path to the dataset directory",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Which split to load from dataset.json",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Case index within the chosen split",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/step1_mid_slice.png"),
        help="Where to save the middle-slice visualization PNG",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show an interactive window instead of saving a PNG (requires GUI backend)",
    )

    args = parser.parse_args()

    image_path, label_path = get_case_paths(args.dataset_root, args.split, args.index)
    if not image_path.exists():
        print(f"Error: Image file not found at {image_path}")
        return
    if label_path is not None and not label_path.exists():
        print(f"Error: Label file not found at {label_path}")
        return
    print(f"Found image: {image_path}")
    if label_path is not None:
        print(f"Found label: {label_path}")

    ct_image = nib.load(str(image_path))
    describe_nifti(ct_image, "CT")
    ct_data = ct_image.get_fdata(dtype=np.float32)
    print(f"  CT intensity stats (float32): min={ct_data.min():.1f} max={ct_data.max():.1f} mean={ct_data.mean():.1f}")

    mask = None
    if label_path is not None and label_path.exists():
        mask_image = nib.load(str(label_path))
        describe_nifti(mask_image, "Mask")
        mask_data = mask_image.get_fdata(dtype=np.float32)
        mask_u8 = (mask_data > 0.5).astype(np.uint8)
        print(f"  mask unique values: {np.unique(mask_u8).tolist()}")
        mask = mask_u8

    saved_path = render_middle_slice(ct_data, mask, args.output, show=args.show)
    if saved_path:
        print(f"Saved middle slice visualization to {saved_path}")

if __name__ == "__main__":
    main()