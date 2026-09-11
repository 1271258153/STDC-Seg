#!/usr/bin/env python3
"""Benchmark STDC1-Seg under one fixed protocol.

Fixed protocol
--------------
* Input: 1 x 3 x 640 x 640, FP32
* Device: the first visible CUDA device (use CUDA_VISIBLE_DEVICES to select it)
* Warm-up: 100 forward passes
* Timing: 500 forward passes, batch size 1
* Synchronization: torch.cuda.synchronize() immediately before and after every
  timed forward pass
* Complexity: THOP MACs converted with 1 MAC = 2 FLOPs
* Output: benchmark_results.csv

Only checkpoint paths are configurable. Checkpoints are optional because
parameters and FLOPs depend on the model structure, not trained values. When a
checkpoint is supplied, only the inference-path tensors are loaded.

The benchmark intentionally excludes the training-only auxiliary segmentation
heads and boundary heads. Its forward path is the same path used to produce
``net(input)[0]`` during evaluation: context path -> feature fusion -> main
segmentation head -> bilinear upsampling.

Example:
    python tools/benchmark_ablation.py
"""

import argparse
import copy
import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from models.model_stages import BiSeNet  # noqa: E402


INPUT_SHAPE = (1, 3, 640, 640)
NUM_CLASSES = 10
WARMUP_RUNS = 100
TIMED_RUNS = 500
CSV_PATH = Path("benchmark_results.csv")


@dataclass(frozen=True)
class ModelSpec:
    name: str
    use_ema: bool
    checkpoint_arg: str


MODEL_SPECS: Tuple[ModelSpec, ...] = (
    ModelSpec("STDC1-Seg", True, "ema_checkpoint"),
)


class STDCInferenceModel(nn.Module):
    """Keep only the modules executed by the main inference output."""

    def __init__(self, use_ema: bool) -> None:
        super().__init__()
        full_model = BiSeNet(
            backbone="STDCNet813",
            n_classes=NUM_CLASSES,
            use_boundary_2=False,
            use_boundary_4=False,
            use_boundary_8=False,
            use_boundary_16=False,
            use_conv_last=False,
            use_ema=use_ema,
        )
        self.cp = full_model.cp
        self.ffm = full_model.ffm
        self.conv_out = full_model.conv_out

        # The backbone also registers ImageNet-classification modules and the
        # original aggregate ``features`` container. They are not executed by
        # STDC-Seg inference; stage containers x2/x4/x8/x16/x32 own the same
        # feature blocks needed by ``forward``. Unregister the unused modules so
        # Params, GFLOPs and FPS all describe exactly the same execution path.
        backbone = self.cp.backbone
        for module_name in (
            "features", "conv_last", "gap", "fc", "bn", "relu",
            "dropout", "linear",
        ):
            setattr(backbone, module_name, None)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        height, width = inputs.shape[-2:]
        _, _, feature_8, _, context_8, _ = self.cp(inputs)
        fused = self.ffm(feature_8, context_8)
        logits = self.conv_out(fused)
        return F.interpolate(
            logits,
            size=(height, width),
            mode="bilinear",
            align_corners=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark STDC1-Seg. Input size, "
            "batch size, warm-up count, timed runs, precision and CSV path "
            "are fixed."
        )
    )
    parser.add_argument("--ema-checkpoint", type=Path)
    return parser.parse_args()


def build_model(spec: ModelSpec) -> nn.Module:
    return STDCInferenceModel(use_ema=spec.use_ema)


def unwrap_state_dict(checkpoint: object) -> Dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must contain a state-dict-like mapping.")

    state = checkpoint
    for key in ("state_dict", "model_state_dict", "model", "net"):
        value = state.get(key)
        if isinstance(value, dict):
            state = value
            break

    if not state or not all(isinstance(key, str) for key in state):
        raise ValueError("No valid model state_dict was found in the checkpoint.")
    return state


def key_candidates(key: str) -> List[str]:
    """Return key variants for raw, DataParallel and wrapped checkpoints."""
    candidates = [key]
    current = key
    prefixes = ("module.", "model.")
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if current.startswith(prefix):
                current = current[len(prefix):]
                candidates.append(current)
                changed = True
                break
    return candidates


def load_checkpoint(model: nn.Module, path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError("Checkpoint does not exist: {}".format(path))

    raw_checkpoint = torch.load(str(path), map_location="cpu")
    raw_state = unwrap_state_dict(raw_checkpoint)
    model_state = model.state_dict()
    matched: Dict[str, torch.Tensor] = {}

    for raw_key, value in raw_state.items():
        if not isinstance(value, torch.Tensor):
            continue
        for candidate in key_candidates(raw_key):
            if candidate in model_state and value.shape == model_state[candidate].shape:
                matched[candidate] = value
                break

    if not matched:
        raise RuntimeError(
            "No checkpoint tensors matched the inference model for: {}".format(path)
        )

    incompatible = model.load_state_dict(matched, strict=False)
    print(
        "  Loaded checkpoint: {} (matched {}/{}, missing {}, ignored {})".format(
            path,
            len(matched),
            len(model_state),
            len(incompatible.missing_keys),
            len(raw_state) - len(matched),
        )
    )


def count_complexity(
    model: nn.Module, input_tensor: torch.Tensor
) -> Tuple[float, float]:
    try:
        from thop import profile
    except ImportError as exc:
        raise ImportError(
            "THOP is required for complexity measurement. Install it with: "
            "pip install thop"
        ) from exc

    params = sum(parameter.numel() for parameter in model.parameters()) / 1e6

    # THOP reports multiply-accumulate operations (MACs). Match the reference
    # benchmark's arithmetic convention: one MAC is two floating-point ops.
    # STDCNet exposes the same backbone blocks through ``features`` and the
    # x2/x4/x8/x16/x32 stage containers. Some THOP versions leave hooks behind
    # when modules are shared this way, so profile an isolated copy and keep the
    # latency model untouched.
    profile_model = copy.deepcopy(model)
    raw_macs, _ = profile(profile_model, inputs=(input_tensor,), verbose=False)
    del profile_model
    gflops = (2.0 * float(raw_macs)) / 1e9
    return params, gflops


def measure_latency(
    model: nn.Module, input_tensor: torch.Tensor
) -> Tuple[float, float]:
    model.eval()
    elapsed_seconds: List[float] = []

    with torch.inference_mode():
        for _ in range(WARMUP_RUNS):
            model(input_tensor)
        torch.cuda.synchronize()

        for _ in range(TIMED_RUNS):
            torch.cuda.synchronize()
            start = time.perf_counter()
            model(input_tensor)
            torch.cuda.synchronize()
            elapsed_seconds.append(time.perf_counter() - start)

    latency_ms = (sum(elapsed_seconds) / TIMED_RUNS) * 1000.0
    fps = 1000.0 / latency_ms
    return latency_ms, fps


def print_table(rows: List[Dict[str, object]]) -> None:
    headers = ("Model", "Params(M)", "GFLOPs", "Latency(ms)", "FPS")
    formatted = [
        (
            str(row["Model"]),
            "{:.4f}".format(row["Params(M)"]),
            "{:.3f}".format(row["GFLOPs"]),
            "{:.3f}".format(row["Latency(ms)"]),
            "{:.2f}".format(row["FPS"]),
        )
        for row in rows
    ]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in formatted))
        for index in range(len(headers))
    ]

    def line(values: Tuple[str, ...]) -> str:
        return " | ".join(
            value.ljust(widths[index]) for index, value in enumerate(values)
        )

    print("\n" + line(headers))
    print("-+-".join("-" * width for width in widths))
    for row in formatted:
        print(line(row))


def save_csv(rows: List[Dict[str, object]]) -> None:
    fieldnames = ["Model", "Params(M)", "GFLOPs", "Latency(ms)", "FPS"]
    with CSV_PATH.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "Model": row["Model"],
                    "Params(M)": "{:.4f}".format(row["Params(M)"]),
                    "GFLOPs": "{:.6f}".format(row["GFLOPs"]),
                    "Latency(ms)": "{:.6f}".format(row["Latency(ms)"]),
                    "FPS": "{:.6f}".format(row["FPS"]),
                }
            )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for latency and FPS testing.")

    torch.manual_seed(304)
    torch.cuda.manual_seed_all(304)
    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    device = torch.device("cuda:0")
    input_tensor = torch.randn(INPUT_SHAPE, device=device, dtype=torch.float32)

    print("PyTorch: {}".format(torch.__version__))
    print("CUDA runtime: {}".format(torch.version.cuda))
    print("GPU: {}".format(torch.cuda.get_device_name(device)))
    print("Input: {}, FP32, batch size 1".format(INPUT_SHAPE))
    print("Warm-up: {}; timed runs: {}".format(WARMUP_RUNS, TIMED_RUNS))
    print("GFLOPs: THOP MACs x 2 (1 MAC = 2 FLOPs)")
    print("Scope: inference path only; auxiliary and boundary heads excluded")

    results: List[Dict[str, object]] = []
    for index, spec in enumerate(MODEL_SPECS, start=1):
        print("\n[{}/{}] {}".format(index, len(MODEL_SPECS), spec.name))
        model = build_model(spec)

        checkpoint: Optional[Path] = getattr(args, spec.checkpoint_arg)
        if checkpoint is not None:
            load_checkpoint(model, checkpoint)
        else:
            print("  No checkpoint supplied; using initialized weights.")

        model.eval().to(device)
        params_m, gflops = count_complexity(model, input_tensor)
        latency_ms, fps = measure_latency(model, input_tensor)
        results.append(
            {
                "Model": spec.name,
                "Params(M)": params_m,
                "GFLOPs": gflops,
                "Latency(ms)": latency_ms,
                "FPS": fps,
            }
        )

        del model
        torch.cuda.empty_cache()

    print_table(results)
    save_csv(results)
    print("\nSaved CSV to: {}".format(CSV_PATH.resolve()))


if __name__ == "__main__":
    main()
