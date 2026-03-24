"""FastAPI serving app — SpleenSeg 2.5D CT spleen segmentation.

Environment variables (all optional, sensible defaults for Docker):
  ONNX_MODEL_PATH   path to unet25d.onnx           (default /models/unet25d.onnx)
  SAMPLES_DIR       root of Task09_Spleen dataset   (default /samples)
  RESULTS_DIR       where to cache inference outputs (default /results)
  NUM_SLICES        2.5D context depth               (default 5)
  THRESHOLD         sigmoid threshold for mask       (default 0.5)
  ROOT_PATH         ASGI root_path for nginx proxy   (default "", set to "/spleenseg/api" in prod)

Run locally (dev):
  uvicorn SpleenSeg.serving.app:app --reload --port 8000
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from SpleenSeg.inference.run_inference_25d import (
    _dice,
    _extract_patch_chw,
    _save_nifti,
    _save_qc_images,
    _sigmoid,
    _stack_slices,
    _tile_starts,
)
from SpleenSeg.preprocessing.transforms import PreprocessConfig, build_preprocessing_transforms

log = logging.getLogger("spleenseg.api")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

# ── Configuration from environment ───────────────────────────────────────────
ONNX_PATH   = Path(os.environ.get("ONNX_MODEL_PATH", "/models/unet25d.onnx"))
SAMPLES_DIR = Path(os.environ.get("SAMPLES_DIR",     "/samples"))
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR",     "/results"))
NUM_SLICES  = int(os.environ.get("NUM_SLICES",  "5"))
THRESHOLD   = float(os.environ.get("THRESHOLD", "0.5"))
# Set to "/spleenseg/api" in production (nginx reverse proxy strips this prefix).
# Leave empty ("") for local dev so uvicorn serves at /.
ROOT_PATH   = os.environ.get("ROOT_PATH", "")

_CFG = PreprocessConfig()  # HU [-200,300], 1.5 mm isotropic, RAS, 128³

# Pre-loaded sample cases.  Paths are relative to SAMPLES_DIR.
SAMPLES: dict[str, dict[str, Any]] = {
    "spleen_12": {
        "image": "imagesTr/spleen_12.nii.gz",
        "label": "labelsTr/spleen_12.nii.gz",
        "description": "High-quality case — expected Dice ≈ 0.968",
    },
    "spleen_20": {
        "image": "imagesTr/spleen_20.nii.gz",
        "label": "labelsTr/spleen_20.nii.gz",
        "description": "Typical case — expected Dice ≈ 0.955",
    },
    "spleen_6": {
        "image": "imagesTr/spleen_6.nii.gz",
        "label": "labelsTr/spleen_6.nii.gz",
        "description": "Challenging case — expected Dice ≈ 0.935",
    },
}

# ── App-level state ───────────────────────────────────────────────────────────
_sess: Any = None             # ONNXRuntime InferenceSession, loaded once
_semaphore: asyncio.Semaphore | None = None   # limits to 1 concurrent inference
_stats: dict[str, Any] = {
    "requests_completed": 0,
    "dice_scores": [],
    "latencies_s": [],
}
_stats_lock = threading.Lock()
_started_at = time.time()

# Queues for SSE streaming — keyed by case_id, holds asyncio.Queue objects
_stream_queues: dict[str, asyncio.Queue] = {}
_stream_queues_lock = threading.Lock()

# ── Static files and templates ────────────────────────────────────────────────
_HERE = Path(__file__).parent
_STATIC_DIR    = _HERE / "static"
_TEMPLATES_DIR = _HERE / "templates"


# ── Lifespan: startup / shutdown ──────────────────────────────────────────────
@asynccontextmanager
async def _lifespan(app: FastAPI):
    global _sess, _semaphore

    _semaphore = asyncio.Semaphore(1)   # one inference at a time on a 4 GB server
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if ONNX_PATH.exists():
        import onnxruntime as ort
        _sess = ort.InferenceSession(
            str(ONNX_PATH), providers=["CPUExecutionProvider"]
        )
        log.info("ONNX session ready: %s", ONNX_PATH)
    else:
        log.warning(
            "ONNX model not found at %s — /run endpoints will return 503", ONNX_PATH
        )

    yield   # app runs

    log.info("Shutting down SpleenSeg API")


# ── App factory ───────────────────────────────────────────────────────────────
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

app = FastAPI(
    title="SpleenSeg",
    description=(
        "**2.5D CT spleen segmentation** — MSD Decathlon Task09_Spleen.\n\n"
        "Choose a pre-loaded sample case from `/samples`, then call "
        "`POST /run/{case_id}` to run live ONNX inference on the server.\n\n"
        "Results include Dice score, inference time, and per-slice QC images "
        "(CT | GT cyan | Prediction magenta | TP/FP/FN comparison)."
    ),
    version="0.1.0",
    # Tells FastAPI it's mounted at /spleenseg/api on the public domain,
    # so Swagger UI links and redirects work correctly behind nginx.
    root_path=ROOT_PATH,
    lifespan=_lifespan,
)

app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


# ── SSE helpers ──────────────────────────────────────────────────────────────
def _publish(case_id: str, step: str, msg: str, done: bool = False, extra: dict | None = None) -> None:
    """Push a JSON event to all SSE queues registered for case_id (thread-safe)."""
    payload: dict[str, Any] = {"step": step, "msg": msg, "done": done}
    if extra:
        payload.update(extra)
    data = json.dumps(payload)
    with _stream_queues_lock:
        queues = list(_stream_queues.get(case_id, {}).values()) if case_id in _stream_queues else []
    for q in queues:
        try:
            q.put_nowait(data)
        except asyncio.QueueFull:
            pass


# ── Core inference (blocking — runs in a thread via asyncio.to_thread) ─────────
def _run_inference_sync(case_id: str) -> dict[str, Any]:
    """Run full preprocessing + ONNX inference for one sample case.

    Saves mask, QC images, and a summary.json to RESULTS_DIR/case_id/.
    Returns the summary dict.
    """
    sample     = SAMPLES[case_id]
    image_path = SAMPLES_DIR / sample["image"]
    label_path = SAMPLES_DIR / sample["label"]

    for p in (image_path, label_path):
        if not p.exists():
            raise FileNotFoundError(f"Sample data not found on server: {p}")

    t0 = time.time()
    _publish(case_id, "start", f"Starting inference for {case_id}")

    # Preprocessing — identical to training: label-crop → 128³ ROI
    _publish(case_id, "preprocess", "Reorienting to RAS, resampling to 1.5 mm isotropic, HU windowing [-200, 300]…")
    tfm = build_preprocessing_transforms(_CFG)
    out = tfm({"image": str(image_path), "label": str(label_path)})

    # Extract numpy volumes [X, Y, Z]
    def _to_numpy(tensor: Any) -> np.ndarray:
        if hasattr(tensor, "detach"):
            return tensor.detach().cpu().numpy()
        return np.asarray(tensor)

    vol       = _to_numpy(out["image"])[0].astype(np.float32, copy=False)
    true_mask = (_to_numpy(out["label"])[0] > 0).astype(np.uint8, copy=False)

    # Try to recover affine from MONAI metadata (for a valid NIfTI)
    affine = None
    meta = out.get("image_meta_dict", {})
    if isinstance(meta, dict) and "affine" in meta:
        try:
            affine = np.asarray(meta["affine"], dtype=np.float32)
        except Exception:
            affine = None

    x, y, z_total = vol.shape
    _publish(case_id, "preprocess", f"Volume shape after preprocessing: {x}×{y}×{z_total} voxels")
    tile   = int(_CFG.roi_size[0])   # 128
    stride = tile - 32               # overlap = 32 px
    xs = _tile_starts(x, tile=tile, stride=stride)
    ys = _tile_starts(y, tile=tile, stride=stride)

    # Load ONNX session info once
    inp_name   = _sess.get_inputs()[0].name
    out_name   = _sess.get_outputs()[0].name
    inp_shape  = _sess.get_inputs()[0].shape
    _publish(
        case_id, "onnx_load",
        f"ONNX model ready — input {inp_name} {inp_shape} → {out_name}, "
        f"provider: {_sess.get_providers()[0]}"
    )

    # Slice-by-slice ONNX inference with tile accumulation
    logits_vol = np.zeros((x, y, z_total), dtype=np.float32)

    for zi in range(z_total):
        stack      = _stack_slices(vol, z_index=zi, num_slices=NUM_SLICES)
        sum_logits = np.zeros((x, y), dtype=np.float32)
        sum_w      = np.zeros((x, y), dtype=np.float32)

        if zi % max(1, z_total // 10) == 0:
            pct = int(100 * zi / z_total)
            _publish(
                case_id, "inference",
                f"ONNX runtime inference — slice {zi}/{z_total} ({pct}%)",
                extra={"progress": pct},
            )
        for x0 in xs:
            for y0 in ys:
                patch, xslice, yslice = _extract_patch_chw(
                    stack, x0=x0, y0=y0, tile=tile
                )
                inp    = patch[None, ...].astype(np.float32, copy=False)  # [1,C,H,W]
                logits = _sess.run(None, {"image": inp})[0][0, 0]         # [H, W]
                logits = logits[
                    : xslice.stop - xslice.start,
                    : yslice.stop - yslice.start,
                ]
                sum_logits[xslice, yslice] += logits
                sum_w[xslice, yslice]      += 1.0

        logits_vol[:, :, zi] = sum_logits / np.maximum(sum_w, 1.0)

    mask  = (_sigmoid(logits_vol) >= THRESHOLD).astype(np.uint8)
    dice  = float(_dice(mask, true_mask))
    elapsed = time.time() - t0
    _publish(case_id, "inference", f"ONNX runtime inference — slice {z_total}/{z_total} (100%)", extra={"progress": 100})

    # Persist results
    out_dir = RESULTS_DIR / case_id
    out_dir.mkdir(parents=True, exist_ok=True)

    mask_path = out_dir / "pred_mask.nii.gz"
    _publish(case_id, "save", "Saving prediction mask (NIfTI) and QC images…")
    _save_nifti(mask_path, mask, affine=affine)

    qc_dir = out_dir / "qc"
    _save_qc_images(
        qc_dir,
        image_xyz=vol,
        mask_xyz=mask,
        title=f"{case_id} | Dice={dice:.4f}",
        true_mask_xyz=true_mask,
    )

    result: dict[str, Any] = {
        "case_id":          case_id,
        "description":      sample["description"],
        "dice":             round(dice, 4),
        "mask_sum":         int(mask.sum()),
        "inference_time_s": round(elapsed, 2),
        "n_qc_slices":      len(list(qc_dir.glob("slice_*.png"))),
        "links": {
            "result":      f"/result/{case_id}",
            "qc_summary":  f"/result/{case_id}/qc/summary",
            "mask":        f"/result/{case_id}/mask",
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2))

    log.info(
        "Inference done: case=%s dice=%.4f time=%.1fs", case_id, dice, elapsed
    )
    _publish(
        case_id, "done",
        f"Done — Dice: {dice:.4f} | Time: {elapsed:.1f}s",
        done=True,
        extra={"dice": round(dice, 4), "inference_time_s": round(elapsed, 2)},
    )

    with _stats_lock:
        _stats["requests_completed"] += 1
        _stats["dice_scores"].append(dice)
        _stats["latencies_s"].append(elapsed)

    return result


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def index(request: Request):
    """Serve the interactive web UI."""
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/run/{case_id}/stream", summary="Server-Sent Events: live inference log", tags=["inference"])
async def run_inference_stream(case_id: str):
    """Streams JSON-encoded log events via SSE while inference runs.
    Connect before (or just as) you call `POST /run/{case_id}`.
    Each event has: step, msg, done, and optionally progress/dice/inference_time_s."""
    if case_id not in SAMPLES:
        raise HTTPException(404, detail=f"Unknown case '{case_id}'.")
    if _sess is None:
        raise HTTPException(503, detail="ONNX model not loaded.")

    # Create a queue for this SSE connection
    q: asyncio.Queue = asyncio.Queue(maxsize=64)
    conn_id = id(q)
    with _stream_queues_lock:
        if case_id not in _stream_queues:
            _stream_queues[case_id] = {}
        _stream_queues[case_id][conn_id] = q

    async def _event_generator():
        try:
            while True:
                try:
                    data = await asyncio.wait_for(q.get(), timeout=120.0)
                except TimeoutError:
                    yield "data: {\"step\":\"timeout\",\"msg\":\"No update in 120 s\",\"done\":true}\n\n"
                    break
                yield f"data: {data}\n\n"
                payload = json.loads(data)
                if payload.get("done"):
                    break
        finally:
            with _stream_queues_lock:
                _stream_queues[case_id].pop(conn_id, None)

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/health", summary="Liveness check", tags=["meta"])
def health():
    """Returns 200 while the server is up. Used by the CD pipeline and load balancer."""
    return {
        "status":       "ok",
        "model_loaded": _sess is not None,
        "uptime_s":     round(time.time() - _started_at, 1),
    }


@app.get("/model-info", summary="Model metadata and request statistics", tags=["meta"])
def model_info():
    """Architecture details, preprocessing config, and rolling request stats."""
    with _stats_lock:
        scores = list(_stats["dice_scores"])
        lats   = list(_stats["latencies_s"])
        count  = int(_stats["requests_completed"])

    return {
        "model":        str(ONNX_PATH),
        "model_loaded": _sess is not None,
        "architecture": "2D UNet — 2.5D input (5 stacked axial slices), ~1.6 M params",
        "num_slices":   NUM_SLICES,
        "threshold":    THRESHOLD,
        "preprocessing": {
            "hu_window":    [_CFG.hu_min, _CFG.hu_max],
            "spacing_mm":   list(_CFG.target_spacing),
            "roi_size":     list(_CFG.roi_size),
            "orientation":  _CFG.axcodes,
        },
        "stats": {
            "requests_completed": count,
            "mean_dice":          round(float(np.mean(scores)), 4) if scores else None,
            "mean_latency_s":     round(float(np.mean(lats)), 2)   if lats   else None,
        },
    }


@app.get("/samples", summary="List available pre-loaded sample cases", tags=["inference"])
def list_samples():
    """Shows which sample cases are available, whether their data exists on the server,
    and whether a cached inference result is already ready."""
    out = []
    for case_id, meta in SAMPLES.items():
        out.append({
            "case_id":       case_id,
            "description":   meta["description"],
            "data_available": (SAMPLES_DIR / meta["image"]).exists(),
            "result_cached":  (RESULTS_DIR / case_id / "summary.json").exists(),
            "run_url":        f"/run/{case_id}",
        })
    return {"samples": out}


@app.post("/run/{case_id}", summary="Run inference on a pre-loaded sample case", tags=["inference"])
async def run_inference(case_id: str):
    """Runs full preprocessing + ONNX inference for the requested case.

    Always runs fresh inference — results are overwritten each call.
    Only one inference runs at a time (server RAM budget).
    """
    if case_id not in SAMPLES:
        raise HTTPException(
            404,
            detail=f"Unknown case '{case_id}'. See /samples for available cases.",
        )
    if _sess is None:
        raise HTTPException(503, detail="ONNX model not loaded — check server logs.")

    # Acquire semaphore — ensures only 1 inference at a time on the 4 GB server
    async with _semaphore:
        try:
            result = await asyncio.to_thread(_run_inference_sync, case_id)
        except FileNotFoundError as exc:
            raise HTTPException(404, detail=str(exc))
        except Exception as exc:
            log.exception("Inference failed for case %s", case_id)
            raise HTTPException(500, detail=f"Inference error: {exc}")

    return result


@app.get("/result/{case_id}", summary="Get cached result JSON", tags=["results"])
def get_result(case_id: str):
    """Returns the stored inference summary (Dice, mask_sum, timing, links)."""
    p = RESULTS_DIR / case_id / "summary.json"
    if not p.exists():
        raise HTTPException(
            404,
            detail=f"No cached result for '{case_id}'. Call POST /run/{case_id} first.",
        )
    return json.loads(p.read_text())


@app.get(
    "/result/{case_id}/qc/summary",
    summary="QC summary mosaic (10 representative slices)",
    tags=["results"],
)
def get_qc_summary(case_id: str):
    """Returns the summary mosaic PNG: 10 evenly-spaced slices, each as a 2×2 grid
    (CT | GT cyan | Prediction magenta | TP/FP/FN comparison)."""
    p = RESULTS_DIR / case_id / "qc" / "summary.png"
    if not p.exists():
        raise HTTPException(404, detail="QC images not found. Run inference first.")
    return FileResponse(str(p), media_type="image/png")


@app.get(
    "/result/{case_id}/qc/slices",
    summary="List available QC slice indices",
    tags=["results"],
)
def list_qc_slices(case_id: str):
    """Returns the sorted list of axial Z indices that have a QC PNG available."""
    qc_dir = RESULTS_DIR / case_id / "qc"
    if not qc_dir.exists():
        raise HTTPException(404, detail="QC images not found. Run inference first.")
    indices = sorted(
        int(p.stem.split("_")[1])
        for p in qc_dir.glob("slice_*.png")
    )
    return {"case_id": case_id, "slices": indices}


@app.get(
    "/result/{case_id}/qc/{z}",
    summary="QC image for a specific axial slice",
    tags=["results"],
)
def get_qc_slice(case_id: str, z: int):
    """Returns the 2×2 QC PNG for axial slice z.
    Valid range depends on the foreground bounding box of the case."""
    p = RESULTS_DIR / case_id / "qc" / f"slice_{z:03d}.png"
    if not p.exists():
        raise HTTPException(404, detail=f"Slice z={z:03d} not found for case '{case_id}'.")
    return FileResponse(str(p), media_type="image/png")


@app.get(
    "/result/{case_id}/mask",
    summary="Download predicted mask as NIfTI (.nii.gz)",
    tags=["results"],
)
def download_mask(case_id: str):
    """Downloads the binary segmentation mask in the preprocessed coordinate space
    (128³, 1.5 mm isotropic). Load with 3D Slicer or ITK-SNAP."""
    p = RESULTS_DIR / case_id / "pred_mask.nii.gz"
    if not p.exists():
        raise HTTPException(404, detail="Mask not found. Run inference first.")
    return FileResponse(
        str(p),
        media_type="application/gzip",
        filename=f"{case_id}_pred_mask.nii.gz",
    )
