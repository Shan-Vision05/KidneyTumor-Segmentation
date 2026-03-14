from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.networks.layers import Norm
from monai.networks.nets import UNet
from monai.transforms import AsDiscrete, Compose, EnsureType, Activations
from torch.utils.data import DataLoader

from SpleenSeg.preprocessing.transforms import PreprocessConfig
from SpleenSeg.training.dataset_25d import DecathlonSpleen25DDataset, Slice25DConfig


def _split_train_val(n_cases: int, val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    """Randomly split case indices into train/val sets."""
    if n_cases <= 0:
        return [], []
    if not (0.0 < val_fraction < 1.0):
        raise ValueError("val_fraction must be in [0.0, 1.0)")

    if n_cases == 1:
        return [0], []

    indices = np.arange(n_cases, dtype=int)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    n_val = max(1, int(round(n_cases * val_fraction)))
    n_val = min(n_val, n_cases - 1)  # ensure at least 1 train
    train_idx = indices[n_val:].tolist()
    val_idx = indices[:n_val].tolist()
    return train_idx, val_idx

def _evaluate_dice(model: torch.nn.Module, dataloader: DataLoader, device: torch.device) -> float:
    post_pred = Compose([EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    post_label = Compose([EnsureType(), AsDiscrete(threshold=0.5)])
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    model.eval()
    dice_metric.reset()

    n_batches = 0
    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            outputs = model(images)
            outputs = post_pred(outputs)
            labels = post_label(labels)
            dice_metric(y_pred=outputs, y=labels)
            n_batches += 1
    if n_batches == 0:
        return float("nan")
    mean_dice = dice_metric.aggregate()
    if mean_dice is None:
        return float("nan")
    return float(mean_dice.item()) if hasattr(mean_dice, "item") else float(mean_dice)

def _save_checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, best_dice: float, args: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "best_dice": best_dice,
            "args": args,
        },
        path,
    )

def _to_mlflow_param_value(value: Any) -> str | int | float | bool:
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if value is None:
        return "None"
    return str(value)

def _mlflow_log_dict(mlflow_module: Any, d:dict, artifact_file: str) -> None:
    if hasattr(mlflow_module, "log_dict"):
        mlflow_module.log_dict(d, artifact_file)
        return

    tmp = Path("artifacts/.mlflow_tmp")
    tmp.mkdir(parents=True, exist_ok=True)
    out_path = tmp / artifact_file
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(d, indent=2, sort_keys=True))
    mlflow_module.log_artifact(str(out_path), artifact_path=str(Path(artifact_file).parent))



def main() -> None:
    parser = argparse.ArgumentParser(description="Step 4: Train 2D UNet on 2.5D slice-stack dataset.")
    parser.add_argument("--dataset-root", type=Path, default=Path("Task09_Spleen"))

    # Data
    parser.add_argument("--num-slices", type=int, default=5)
    parser.add_argument("--negative-ratio", type=float, default=1.0)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-cases", type=int, default=0, help="If >0, limit total cases used before splitting")

    # Preprocess
    parser.add_argument("--hu-min", type=float, default=-200.0)
    parser.add_argument("--hu-max", type=float, default=300.0)
    parser.add_argument("--spacing", type=float, nargs=3, default=(1.5, 1.5, 1.5))
    parser.add_argument("--roi-size", type=int, nargs=3, default=(128, 128, 128))
    parser.add_argument("--axcodes", type=str, default="RAS")

    # Train
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=50, help="Print a batch update every N steps (0 disables)")
    parser.add_argument("--progress", action="store_true", help="Show tqdm progress bars (requires tqdm)")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--amp", action="store_true", help="Use mixed precision (CUDA only)")

    # Output
    parser.add_argument("--run-dir", type=Path, default=Path("artifacts/runs/unet25d"))

    # MLflow
    parser.add_argument("--mlflow", action="store_true", help="Enable MLflow tracking")
    parser.add_argument("--mlflow-tracking-uri", type=str, default="", help="e.g. file:./mlruns or http://localhost:5000")
    parser.add_argument("--mlflow-experiment", type=str, default="SpleenSeg", help="MLflow experiment name")
    parser.add_argument("--mlflow-run-name", type=str, default="", help="Optional run name")
    parser.add_argument("--mlflow-log-checkpoints", action="store_true", help="Log best/last checkpoints as MLflow artifacts")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(args.amp) and device.type == "cuda"

    pre_cfg = PreprocessConfig(
        hu_min=args.hu_min,
        hu_max=args.hu_max,
        target_spacing=tuple(args.spacing),
        roi_size=tuple(args.roi_size),
        axcodes=args.axcodes,
    )
    slice_cfg = Slice25DConfig(
        num_slices=args.num_slices,
        positive_only=True,
        negative_ratio=float(args.negative_ratio),
    )

    dataset_json = args.dataset_root / "dataset.json"
    if not dataset_json.exists():
        raise FileNotFoundError(f"Missing dataset.json at: {dataset_json}")
    info = json.loads(dataset_json.read_text())
    n_cases = len(info.get("training", []))
    if n_cases <=0:
        raise ValueError("No training cases found in dataset.json")
    
    if args.max_cases and args.max_cases > 0:
        n_cases = min(n_cases, args.max_cases)
        print(f"Limiting to max_cases={n_cases} for quick verification")
    
    train_case_idx, val_case_idx = _split_train_val(n_cases=n_cases, val_fraction=args.val_fraction, seed=args.seed)

    print(f"Device: {device} (AMP={use_amp})")
    print(
        f"Cases: total={n_cases} train={len(train_case_idx)} val={len(val_case_idx)} "
        f"(val_fraction={args.val_fraction})"
    )
    print(f"Case indices: train={train_case_idx} val={val_case_idx}")

    train_ds = DecathlonSpleen25DDataset(
        dataset_root=args.dataset_root,
        preprocess_config=pre_cfg,
        slice_config=slice_cfg,
        augment=True,
        case_indices=train_case_idx,
        verbose=False,
        seed=int(args.seed),
    )
    val_ds = DecathlonSpleen25DDataset(
        dataset_root=args.dataset_root,
        preprocess_config=pre_cfg,
        slice_config=slice_cfg,
        augment=False,
        case_indices=val_case_idx,
        verbose=False,
        seed=int(args.seed),
    )

    print(f"Train dataset size (slices): {len(train_ds)}")
    print(f"Val dataset size (slices): {len(val_ds)}")

    persistent_workers = int(args.num_workers) > 0

    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
        persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
        persistent_workers=persistent_workers,
    )
    
    model = UNet(
        spatial_dims=2,
        in_channels=int(args.num_slices),
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
    ).to(device)

    loss_fn = DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    run_dir = args.run_dir
    ckpt_best = run_dir / "checkpoints" / "best.pt"
    ckpt_last = run_dir / "checkpoints" / "last.pt"

    best_dice = -1.0

    mlflow = None
    mlflow_run_started = False
    if bool(args.mlflow):
        import mlflow as _mlflow

        mlflow = _mlflow
        if str(args.mlflow_tracking_uri).strip():
            mlflow.set_tracking_uri(str(args.mlflow_tracking_uri).strip())
        mlflow.set_experiment(str(args.mlflow_experiment))

        run_name = str(args.mlflow_run_name).strip() or None
        mlflow.start_run(run_name=run_name)
        mlflow_run_started = True

        cli_params = {k: _to_mlflow_param_value(v) for k, v in vars(args).items()}
        for k, v in cli_params.items():
            mlflow.log_param(f"cli.{k}", v)
        for k, v in asdict(pre_cfg).items():
            mlflow.log_param(f"preprocess.{k}", _to_mlflow_param_value(v))
        for k, v in asdict(slice_cfg).items():
            mlflow.log_param(f"slice.{k}", _to_mlflow_param_value(v))

        mlflow.log_param("split.n_cases", int(n_cases))
        mlflow.log_param("split.n_train", int(len(train_case_idx)))
        mlflow.log_param("split.n_val", int(len(val_case_idx)))

        _mlflow_log_dict(
            mlflow,
            {"train": train_case_idx, "val": val_case_idx},
            artifact_file="splits/case_indices.json",
        )

    try:
        for epoch in range(1, int(args.epochs) + 1):
            model.train()
            epoch_loss = 0.0
            n_steps = 0

            train_iter = train_loader
            if bool(args.progress):
                try:
                    from tqdm.auto import tqdm  # type: ignore

                    train_iter = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [train]", leave=False)
                except Exception:
                    train_iter = train_loader

            for batch in train_iter:
                images = batch["image"].to(device)  # [B, C, H, W]
                labels = batch["label"].to(device).float()  # [B, 1, H, W]

                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    outputs = model(images)  # [B, 1, H, W]
                    loss = loss_fn(outputs, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                epoch_loss += float(loss.detach().item())
                n_steps += 1

                if int(args.log_every) > 0 and (n_steps % int(args.log_every) == 0):
                    total = len(train_loader) if hasattr(train_loader, "__len__") else -1
                    if total > 0:
                        print(f"  step {n_steps}/{total} | loss={float(loss.detach().item()):.4f}")
                    else:
                        print(f"  step {n_steps} | loss={float(loss.detach().item()):.4f}")

                if bool(args.progress):
                    try:
                        # tqdm iterator has set_postfix; normal loader doesn't.
                        train_iter.set_postfix(loss=float(loss.detach().item()))  # type: ignore[attr-defined]
                    except Exception:
                        pass

            epoch_loss /= max(1, n_steps)
            print(f"Epoch {epoch}/{args.epochs} | train loss={epoch_loss:.4f}")
            if mlflow is not None:
                mlflow.log_metric("train.loss", float(epoch_loss), step=int(epoch))

            val_dice = _evaluate_dice(model=model, dataloader=val_loader, device=device)
            if np.isnan(val_dice):
                val_dice = None
                print(f"Epoch {epoch}/{args.epochs} | val dice skipped (empty val dataset)")
            else:
                print(f"Epoch {epoch}/{args.epochs} | val dice={val_dice:.4f}")
                if mlflow is not None:
                    mlflow.log_metric("val.dice", float(val_dice), step=int(epoch))

            _save_checkpoint(
                path=ckpt_last,
                model=model,
                optimizer=optimizer,
                epoch=int(epoch),
                best_dice=float(best_dice),
                args={"cli": vars(args), "preprocess": asdict(pre_cfg), "slice": asdict(slice_cfg)},
            )

            if val_dice is not None and float(val_dice) > best_dice:
                best_dice = float(val_dice)
                _save_checkpoint(
                    ckpt_best,
                    model=model,
                    optimizer=optimizer,
                    epoch=int(epoch),
                    best_dice=float(best_dice),
                    args={"cli": vars(args), "preprocess": asdict(pre_cfg), "slice": asdict(slice_cfg)},
                )
                print(f"New best dice={best_dice:.4f} | saved: {ckpt_best}")
                if mlflow is not None:
                    mlflow.log_metric("val.best_dice", float(best_dice), step=int(epoch))

        print(f"Done. Best val dice={best_dice:.4f}")
        print(f"Last checkpoint: {ckpt_last.resolve()}")
        if ckpt_best.exists():
            print(f"Best checkpoint: {ckpt_best.resolve()}")

        if mlflow is not None:
            mlflow.log_metric("val.best_dice_final", float(best_dice))
            if bool(args.mlflow_log_checkpoints):
                if ckpt_last.exists():
                    mlflow.log_artifact(str(ckpt_last), artifact_path="checkpoints")
                if ckpt_best.exists():
                    mlflow.log_artifact(str(ckpt_best), artifact_path="checkpoints")
                cfg = {"cli": vars(args), "preprocess": asdict(pre_cfg), "slice": asdict(slice_cfg)}
                _mlflow_log_dict(mlflow, cfg, artifact_file="config/run_config.json")
    finally:
        if mlflow is not None and mlflow_run_started:
            mlflow.end_run()

if __name__ == "__main__":
    main()