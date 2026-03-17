from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from SpleenSeg.preprocessing.transforms import PreprocessConfig, build_preprocessing_transforms


@dataclass(frozen=True)
class Slice25DConfig:
    num_slices: int = 5  # must be odd
    positive_only: bool = True
    negative_ratio: float = 1.0  # negatives per positive (approx)


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
    name = image_path.name
    return name[: -len(".nii.gz")] if name.endswith(".nii.gz") else image_path.stem


def _stack_slices(vol_xyz: np.ndarray, z_index: int, num_slices: int) -> np.ndarray:
    """Stack k adjacent axial slices (along Z=last axis) as channels.

    vol_xyz: [X, Y, Z]
    returns: [C=num_slices, X, Y]
    """
    if num_slices % 2 != 1:
        raise ValueError("num_slices must be odd")
    half = num_slices // 2
    z = vol_xyz.shape[2]
    zs = [min(max(z_index + dz, 0), z - 1) for dz in range(-half, half + 1)]
    stack = np.stack([vol_xyz[:, :, zi] for zi in zs], axis=0)
    return stack


class DecathlonSpleen25DDataset(Dataset):
    """2.5D slice-stack dataset for Task09_Spleen.

    Each sample is:
      - image: float32 torch tensor [C=num_slices, H, W]
      - label: uint8 torch tensor [1, H, W] (center slice mask)
      - meta: dict with case_id and slice index

    Implementation choice:
      - We apply the 3D preprocessing once per case (cache in RAM), then generate 2.5D
        slices from the cached volumes. This keeps training fast and deterministic.
    """

    def __init__(
        self,
        dataset_root: Path | str,
        preprocess_config: PreprocessConfig | None = None,
        slice_config: Slice25DConfig | None = None,
        augment: bool = False,
        case_indices: Sequence[int] | None = None,
        max_cases: int | None = None,
        verbose: bool = False,
        seed: int = 0,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.preprocess_config = preprocess_config or PreprocessConfig()
        self.slice_config = slice_config or Slice25DConfig()
        self.augment = bool(augment)
        self.case_indices = list(case_indices) if case_indices is not None else None
        self.max_cases = max_cases
        self.verbose = bool(verbose)
        self.rng = np.random.default_rng(seed)

        if self.slice_config.num_slices % 2 != 1:
            raise ValueError("Slice25DConfig.num_slices must be odd")
        if self.slice_config.negative_ratio < 0:
            raise ValueError("Slice25DConfig.negative_ratio must be >= 0")

        self.pairs = _load_training_pairs(self.dataset_root)

        # Optional subset selection (useful for train/val split by case).
        if self.case_indices is not None:
            if not self.case_indices:
                raise ValueError("case_indices was provided but empty")
            n_total = len(self.pairs)
            bad = [i for i in self.case_indices if i < 0 or i >= n_total]
            if bad:
                raise IndexError(f"case_indices contains out-of-range indices: {bad} (n_total={n_total})")
            self.pairs = [self.pairs[i] for i in self.case_indices]

        if self.max_cases is not None:
            if self.max_cases <= 0:
                raise ValueError("max_cases must be > 0 when provided")
            self.pairs = self.pairs[: self.max_cases]
        self.case_ids = [_case_id_from_image_path(p[0]) for p in self.pairs]

        self._pre_tfm = build_preprocessing_transforms(self.preprocess_config)

        # Cache: list of tuples (image_xyz, label_xyz) where each is [X,Y,Z]
        self._cache: list[tuple[np.ndarray, np.ndarray]] = []
        self._index: list[tuple[int, int]] = []  # (case_idx, z)

        self._build_cache_and_index()

        self._aug_tfm = None
        if self.augment:
            self._aug_tfm = _build_augmentation_2d()

    def _build_cache_and_index(self) -> None:
        pos_slices: list[tuple[int, int]] = []
        neg_slices_by_case: list[list[int]] = []

        for case_idx, (image_path, label_path) in enumerate(self.pairs):
            if self.verbose:
                case_id = _case_id_from_image_path(image_path)
                print(f"[cache] preprocessing case {case_idx+1}/{len(self.pairs)}: {case_id}")
            sample = {"image": str(image_path), "label": str(label_path)}
            out = self._pre_tfm(sample)

            image = out["image"]
            label = out["label"]

            # Convert to numpy (works for torch tensors or numpy arrays)
            if hasattr(image, "detach"):
                image_np = image.detach().cpu().numpy()
            else:
                image_np = np.asarray(image)

            if hasattr(label, "detach"):
                label_np = label.detach().cpu().numpy()
            else:
                label_np = np.asarray(label)

            # image_np/label_np: [1, X, Y, Z]
            if image_np.ndim != 4 or label_np.ndim != 4:
                raise ValueError(f"Unexpected tensor dims for case {case_idx}: {image_np.shape} / {label_np.shape}")

            vol = image_np[0].astype(np.float32, copy=False)
            msk = (label_np[0] > 0).astype(np.uint8, copy=False)

            self._cache.append((vol, msk))

            # Foreground-aware slice selection
            fg_per_z = msk.sum(axis=(0, 1))  # [Z]
            pos_zs = np.where(fg_per_z > 0)[0].tolist()
            neg_zs = np.where(fg_per_z == 0)[0].tolist()

            for zi in pos_zs:
                pos_slices.append((case_idx, int(zi)))

            neg_slices_by_case.append([int(zi) for zi in neg_zs])

        if not self.slice_config.positive_only:
            # Include all slices from all cases.
            for case_idx, (_, msk) in enumerate(self._cache):
                for zi in range(msk.shape[2]):
                    self._index.append((case_idx, zi))
            return

        # Positive slices always included.
        self._index.extend(pos_slices)

        # Add a sampled set of negative slices to balance training.
        if self.slice_config.negative_ratio > 0 and len(pos_slices) > 0:
            # Allocate negatives roughly proportional to positives per case.
            pos_count_by_case = [0 for _ in self.pairs]
            for case_idx, _ in pos_slices:
                pos_count_by_case[case_idx] += 1

            for case_idx, neg_zs in enumerate(neg_slices_by_case):
                if not neg_zs:
                    continue
                n_pos = pos_count_by_case[case_idx]
                n_neg = int(round(n_pos * self.slice_config.negative_ratio))
                if n_neg <= 0:
                    continue
                chosen = self.rng.choice(neg_zs, size=min(n_neg, len(neg_zs)), replace=False)
                for zi in chosen.tolist():
                    self._index.append((case_idx, int(zi)))

        # Shuffle index for better mixing across cases.
        self.rng.shuffle(self._index)

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict:
        case_idx, z = self._index[idx]
        case_id = self.case_ids[case_idx]
        vol, msk = self._cache[case_idx]

        image_stack = _stack_slices(vol, z_index=z, num_slices=self.slice_config.num_slices)
        label_2d = msk[:, :, z][None, ...]  # [1, X, Y]

        # Convert to torch tensors. Treat X,Y as H,W.
        image_t = torch.from_numpy(image_stack.astype(np.float32, copy=False))
        label_t = torch.from_numpy(label_2d.astype(np.uint8, copy=False))

        if self._aug_tfm is not None:
            # MONAI-style dict augmentation expects numpy or torch; both work.
            out = self._aug_tfm({"image": image_t, "label": label_t})
            image_t = out["image"]
            label_t = out["label"]

        return {
            "image": image_t,
            "label": label_t,
            "meta": {"case_id": case_id, "z": int(z)},
        }


def _build_augmentation_2d():
    """Lightweight 2D augmentations applied consistently to image+label."""
    from monai.transforms import (
        Compose,
        EnsureTyped,
        RandFlipd,
        RandRotate90d,
        RandScaleIntensityd,
        RandShiftIntensityd,
    )

    return Compose(
        [
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandRotate90d(keys=["image", "label"], prob=0.25, max_k=3),
            RandScaleIntensityd(keys=["image"], prob=0.25, factors=0.1),
            RandShiftIntensityd(keys=["image"], prob=0.25, offsets=0.05),
            EnsureTyped(keys=["image", "label"]),
        ]
    )
