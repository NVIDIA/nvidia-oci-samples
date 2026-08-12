#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unified entry point for the FLUX.1-schnell benchmark runners."""

import argparse
import importlib.util
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


PYTORCH_CONTAINER = "nvcr.io/nvidia/pytorch:26.07-py3"
TRTLLM_CONTAINER = "nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc22"
SGLANG_CONTAINER = "lmsysorg/sglang:v0.5.12"
VLLM_OMNI_CONTAINER = "vllm/vllm-omni:v0.26.0"
REPOSITORY_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ModeSpec:
    label: str
    runner: str
    runner_mode: str
    container: str
    checkpoint: str
    packages: tuple[str, ...]
    cuda_graph: bool = False


MODES = {
    "pytorch-bf16-compile": ModeSpec(
        "BFL PyTorch compile",
        "pytorch",
        "compile",
        PYTORCH_CONTAINER,
        "native",
        ("torch", "flux", "PIL"),
    ),
    "hf-diffusers-bf16-compile": ModeSpec(
        "HF Diffusers compile",
        "hf",
        "bf16-compile",
        TRTLLM_CONTAINER,
        "diffusers",
        ("torch", "diffusers", "transformers"),
    ),
    "torchao-diffusers-bf16-regional": ModeSpec(
        "TorchAO BF16 regional compile",
        "torchao",
        "bf16-regional",
        PYTORCH_CONTAINER,
        "diffusers",
        ("torch", "diffusers", "transformers", "torchao", "mslk"),
    ),
    "torchao-diffusers-nvfp4-regional": ModeSpec(
        "TorchAO NVFP4 regional compile",
        "torchao",
        "nvfp4-regional",
        PYTORCH_CONTAINER,
        "diffusers",
        ("torch", "diffusers", "transformers", "torchao", "mslk"),
    ),
    "torchao-diffusers-nvfp4-regional-cg": ModeSpec(
        "TorchAO NVFP4 regional compile + CUDA Graph",
        "torchao",
        "nvfp4-regional-cg",
        PYTORCH_CONTAINER,
        "diffusers",
        ("torch", "diffusers", "transformers", "torchao", "mslk"),
        cuda_graph=True,
    ),
    "sglang-bf16-offline-compile": ModeSpec(
        "SGLang BF16 offline batch + compile",
        "sglang",
        "bf16-offline-compile",
        SGLANG_CONTAINER,
        "diffusers",
        ("torch", "sglang"),
    ),
    "vllm-omni-bf16-offline": ModeSpec(
        "vLLM Omni BF16 offline batch",
        "vllm_omni",
        "bf16-offline",
        VLLM_OMNI_CONTAINER,
        "diffusers",
        ("torch", "vllm_omni"),
    ),
    "trtllm-visualgen-bf16": ModeSpec(
        "TensorRT-LLM VisualGen BF16",
        "visualgen",
        "bf16",
        TRTLLM_CONTAINER,
        "diffusers",
        ("torch", "tensorrt_llm"),
    ),
    "trtllm-visualgen-bf16-cuda-graph": ModeSpec(
        "TensorRT-LLM VisualGen BF16 + CUDA Graph",
        "visualgen",
        "bf16",
        TRTLLM_CONTAINER,
        "diffusers",
        ("torch", "tensorrt_llm"),
        cuda_graph=True,
    ),
    "trtllm-visualgen-nvfp4": ModeSpec(
        "TensorRT-LLM VisualGen dynamic NVFP4",
        "visualgen",
        "nvfp4",
        TRTLLM_CONTAINER,
        "diffusers",
        ("torch", "tensorrt_llm", "modelopt"),
    ),
    "trtllm-visualgen-nvfp4-cuda-graph": ModeSpec(
        "TensorRT-LLM VisualGen dynamic NVFP4 + CUDA Graph",
        "visualgen",
        "nvfp4",
        TRTLLM_CONTAINER,
        "diffusers",
        ("torch", "tensorrt_llm", "modelopt"),
        cuda_graph=True,
    ),
    "trt-bf16-eager": ModeSpec(
        "TensorRT BF16",
        "trt",
        "bf16:eager",
        PYTORCH_CONTAINER,
        "onnx",
        ("torch", "flux", "PIL", "tensorrt", "polygraphy"),
    ),
    "trt-bf16-cuda-graph": ModeSpec(
        "TensorRT BF16 + CUDA Graph",
        "trt",
        "bf16:cuda_graph",
        PYTORCH_CONTAINER,
        "onnx",
        ("torch", "flux", "PIL", "tensorrt", "polygraphy"),
        cuda_graph=True,
    ),
    "trt-fp4-eager": ModeSpec(
        "TensorRT NVFP4",
        "trt",
        "fp4:eager",
        PYTORCH_CONTAINER,
        "onnx",
        ("torch", "flux", "PIL", "tensorrt", "polygraphy"),
    ),
    "trt-fp4-cuda-graph": ModeSpec(
        "TensorRT NVFP4 + CUDA Graph",
        "trt",
        "fp4:cuda_graph",
        PYTORCH_CONTAINER,
        "onnx",
        ("torch", "flux", "PIL", "tensorrt", "polygraphy"),
        cuda_graph=True,
    ),
}


def required_path(value: Path | None, flag: str) -> Path:
    if value is None:
        raise ValueError(f"{flag} is required for this mode")
    return value


def backend_output_dir(root: Path, spec: ModeSpec) -> Path:
    if spec.runner == "hf":
        return root / "hf-diffusers"
    if spec.runner == "torchao":
        return root / "torchao-diffusers"
    if spec.runner == "visualgen":
        suffix = "-cg" if spec.cuda_graph else ""
        return root / f"trtllm-visualgen{suffix}"
    if spec.runner == "sglang":
        return root / "sglang"
    if spec.runner == "vllm_omni":
        return root / "vllm-omni"
    return root


def build_command(
    args: argparse.Namespace, repository_root: Path = REPOSITORY_ROOT
) -> list[str]:
    spec = MODES[args.mode]
    output_dir = backend_output_dir(args.output_dir, spec)
    common = [
        "--warmup",
        str(args.warmup),
        "--iterations",
        str(args.iterations),
        "--output-dir",
        str(output_dir),
        "--height",
        "1024",
        "--width",
        "1024",
        "--steps",
        "4",
    ]

    if spec.runner == "hf":
        model = required_path(args.model, "--model")
        command = [
            sys.executable,
            "-m",
            "benchmarks.flux1_schnell.hf_diffusers_flux_sweep",
            "--model",
            str(model),
            "--mode",
            spec.runner_mode,
            "--batches",
            *map(str, args.batches),
            "--batch-semantics",
            "request-batch",
            *common,
        ]
    elif spec.runner == "torchao":
        model = required_path(args.model, "--model")
        command = [
            sys.executable,
            "-m",
            "benchmarks.flux1_schnell.torchao_diffusers_flux_sweep",
            "--model",
            str(model),
            "--mode",
            spec.runner_mode,
            "--batches",
            *map(str, args.batches),
            *common,
        ]
    elif spec.runner == "visualgen":
        model = required_path(args.model, "--model")
        config_name = (
            f"visualgen_{spec.runner_mode}{'_cg' if spec.cuda_graph else ''}.yaml"
        )
        command = [
            sys.executable,
            "-m",
            "benchmarks.flux1_schnell.visualgen_flux_sweep",
            "--model",
            str(model),
            "--config",
            str(repository_root / "benchmarks/flux1_schnell/configs" / config_name),
            "--precision",
            spec.runner_mode,
            "--batches",
            *map(str, args.batches),
            "--batch-semantics",
            "request-batch",
            *common,
        ]
    elif spec.runner == "sglang":
        model = required_path(args.model, "--model")
        command = [
            sys.executable,
            "-m",
            "benchmarks.flux1_schnell.sglang_flux_sweep",
            "--model",
            str(model),
            "--batches",
            *map(str, args.batches),
            *common,
        ]
    elif spec.runner == "vllm_omni":
        model = required_path(args.model, "--model")
        command = [
            sys.executable,
            "-m",
            "benchmarks.flux1_schnell.vllm_omni_flux_sweep",
            "--model",
            str(model),
            "--batches",
            *map(str, args.batches),
            *common,
        ]
    elif spec.runner == "pytorch":
        native_model_dir = required_path(args.native_model_dir, "--native-model-dir")
        command = [
            sys.executable,
            "-m",
            "benchmarks.flux1_schnell.flux_batch_sweep",
            "--backend",
            "pytorch",
            "--precision",
            "bf16",
            "--variant",
            spec.runner_mode,
            "--native-model-dir",
            str(native_model_dir),
            "--batch-sizes",
            *map(str, args.batches),
            "--batch-semantics",
            "request-batch",
            *common,
        ]
    else:
        onnx_dir = required_path(args.onnx_dir, "--onnx-dir")
        engine_root = required_path(args.engine_root, "--engine-root")
        precision, variant = spec.runner_mode.split(":", 1)
        command = [
            sys.executable,
            "-m",
            "benchmarks.flux1_schnell.flux_batch_sweep",
            "--backend",
            "trt",
            "--precision",
            precision,
            "--variant",
            variant,
            "--onnx-dir",
            str(onnx_dir),
            "--engine-root",
            str(engine_root),
            "--batch-sizes",
            *map(str, args.batches),
            "--batch-semantics",
            "request-batch",
            *common,
        ]

    if args.nsys_capture:
        command.append("--nsys-capture")
    if args.build_only:
        if spec.runner != "trt":
            raise ValueError("--build-only is supported only by TensorRT modes")
        command.append("--build-only")
    return command


def missing_paths(args: argparse.Namespace, spec: ModeSpec) -> list[str]:
    missing: list[str] = []

    def require(path: Path, kind: str = "path") -> None:
        if not path.exists():
            missing.append(f"missing {kind}: {path}")

    if spec.checkpoint == "diffusers":
        model = required_path(args.model, "--model")
        if not model.is_dir():
            return [f"missing Diffusers checkpoint directory: {model}"]
        for relative in (
            "model_index.json",
            "scheduler",
            "transformer",
            "vae",
            "text_encoder",
            "text_encoder_2",
            "tokenizer",
            "tokenizer_2",
        ):
            require(model / relative)
    elif spec.checkpoint == "native":
        model = required_path(args.native_model_dir, "--native-model-dir")
        if not model.is_dir():
            return [f"missing BFL native checkpoint directory: {model}"]
        require(model / "flux1-schnell.safetensors")
        require(model / "ae.safetensors")
        require(model.parent / "google_t5-v1_1-xxl/pytorch_model.bin")
        require(model.parent / "openai_clip-vit-large-patch14/model.safetensors")
    else:
        onnx_dir = required_path(args.onnx_dir, "--onnx-dir")
        if not onnx_dir.is_dir():
            return [f"missing BFL ONNX checkpoint directory: {onnx_dir}"]
        precision = spec.runner_mode.split(":", 1)[0]
        for relative in (
            "clip.opt/model.onnx",
            "t5.opt/model.onnx",
            "vae.opt/model.onnx",
            f"transformer.opt/{precision}/model.onnx",
        ):
            require(onnx_dir / relative)
        engine_root = required_path(args.engine_root, "--engine-root")
        if not args.build_only:
            for batch in args.batches:
                plan_dir = engine_root / f"b{batch}/flux-schnell"
                pattern = f"transformer_{precision}*.plan"
                if not any(plan_dir.glob(pattern)):
                    missing.append(
                        f"missing TensorRT plan for B{batch}: {plan_dir / pattern}"
                    )
    return missing


def preflight_errors(args: argparse.Namespace) -> list[str]:
    spec = MODES[args.mode]
    errors = [
        f"missing Python module: {package}"
        for package in spec.packages
        if importlib.util.find_spec(package) is None
    ]
    errors.extend(missing_paths(args, spec))
    return errors


def print_modes() -> None:
    mode_width = max(len("MODE"), *(len(key) for key in MODES))
    label_width = max(len("ENGINE"), *(len(spec.label) for spec in MODES.values()))
    print(f"{'MODE':{mode_width}} | {'ENGINE':{label_width}} | CONTAINER")
    for key, spec in MODES.items():
        print(f"{key:{mode_width}} | {spec.label:{label_width}} | {spec.container}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a fair 1024x1024 FLUX.1-schnell benchmark through one entry point"
    )
    parser.add_argument("--list-modes", action="store_true")
    parser.add_argument("--mode", choices=tuple(MODES))
    parser.add_argument("--model", type=Path, help="Diffusers-format checkpoint directory")
    parser.add_argument(
        "--native-model-dir", type=Path, help="BFL native checkpoint directory"
    )
    parser.add_argument("--onnx-dir", type=Path, help="BFL ONNX checkpoint directory")
    parser.add_argument("--engine-root", type=Path, help="TensorRT plan root")
    parser.add_argument("--batches", type=int, nargs="+", default=[1])
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--nsys-capture", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate packages and checkpoints, then exit",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the backend command without checking or running it",
    )
    args = parser.parse_args()
    if args.list_modes:
        print_modes()
        raise SystemExit(0)
    if args.mode is None:
        parser.error("--mode is required unless --list-modes is used")
    if any(batch < 1 or batch > 32 for batch in args.batches):
        parser.error("--batches values must be between 1 and 32")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    return args


def main() -> None:
    args = parse_args()
    spec = MODES[args.mode]
    try:
        command = build_command(args)
    except ValueError as exc:
        raise SystemExit(f"configuration error: {exc}") from exc

    print(f"Mode: {spec.label}")
    print(f"Expected container: {spec.container}")
    print(f"Command: {shlex.join(command)}")
    if args.dry_run:
        return

    try:
        errors = preflight_errors(args)
    except ValueError as exc:
        raise SystemExit(f"preflight failed: {exc}") from exc
    if errors:
        print("Preflight failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        print(f"Use the tested container: {spec.container}", file=sys.stderr)
        raise SystemExit(2)
    print("Preflight: ok")
    if args.check_only:
        return

    if args.nsys_capture:
        sys.stdout.flush()
        sys.stderr.flush()
        os.chdir(REPOSITORY_ROOT)
        os.execv(command[0], command)

    completed = subprocess.run(command, cwd=REPOSITORY_ROOT, check=False)
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
