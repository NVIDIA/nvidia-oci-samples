#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

# Derived in part from the BFL FLUX TensorRT integration. See
# THIRD_PARTY_NOTICES.md for the pinned upstream revision and license.

import gc
import inspect
import os
import shlex
import shutil
import subprocess
from collections import defaultdict
from contextlib import nullcontext
from typing import Any

import tensorrt as trt
import torch
from fire import Fire
from packaging.version import Version
from polygraphy.backend.trt import engine_from_bytes

from flux import cli as flux_cli
from flux.cli import main
from flux.trt.engine.base_engine import Engine, TRT_OFFLOAD_POLICY
from flux.trt.trt_manager import TRTManager
from flux.trt.trt_config.base_trt_config import TRTBaseConfig, trt_version


_original_build_trt_engine = TRTBaseConfig.build_trt_engine
_original_engine_cuda = Engine.cuda
_original_engine_infer = Engine.infer
_original_device_memory_size = Engine.device_memory_size
_original_init_runtime = TRTManager.init_runtime
_original_stop_runtime = TRTManager.stop_runtime
_original_save_image = flux_cli.save_image
_captured_engines = []


def env_enabled(name: str) -> bool:
    return os.getenv(name, "0").lower() in {"1", "true", "yes", "on"}


def build_trt_engine_trt11(
    engine_path: str,
    onnx_path: str,
    strongly_typed: bool = False,
    tf32: bool = True,
    bf16: bool = False,
    fp8: bool = False,
    fp4: bool = False,
    input_profile: dict[str, Any] | None = None,
    update_output_names: list[str] | None = None,
    enable_refit: bool = False,
    enable_all_tactics: bool = False,
    timing_cache: str | None = None,
    native_instancenorm: bool = True,
    builder_optimization_level: int = 3,
    precision_constraints: str = "none",
    verbose: bool = False,
) -> None:
    if Version(trt_version).major < 11:
        _original_build_trt_engine(
            engine_path=engine_path,
            onnx_path=onnx_path,
            strongly_typed=strongly_typed,
            tf32=tf32,
            bf16=bf16,
            fp8=fp8,
            fp4=fp4,
            input_profile=input_profile,
            update_output_names=update_output_names,
            enable_refit=enable_refit,
            enable_all_tactics=enable_all_tactics,
            timing_cache=timing_cache,
            native_instancenorm=native_instancenorm,
            builder_optimization_level=builder_optimization_level,
            precision_constraints=precision_constraints,
            verbose=verbose,
        )
        return

    # TensorRT 11 is always strongly typed and removed the BF16/FP8/FP4/TF32
    # builder flags. The BFL ONNX graphs already carry explicit BF16 tensor and
    # initializer types, so retain those model types and omit obsolete flags.
    polygraphy = shutil.which("polygraphy")
    if polygraphy is None:
        raise RuntimeError("polygraphy entry point is not available")
    command = [
        polygraphy,
        "convert",
        onnx_path,
        "--convert-to",
        "trt",
        "--output",
        engine_path,
    ]
    if enable_refit:
        command.append("--refittable")
    if not enable_all_tactics:
        command.append("--tactic-sources")
    if native_instancenorm:
        command.extend(("--onnx-flags", "native_instancenorm"))
    command.extend(("--builder-optimization-level", str(builder_optimization_level)))
    if timing_cache:
        command.extend(("--load-timing-cache", timing_cache, "--save-timing-cache", timing_cache))
    command.extend(("--verbosity", "extra_verbose" if verbose else "error"))
    if update_output_names:
        command.append("--trt-outputs")
        command.extend(update_output_names)
    if input_profile:
        profile_args: dict[str, list[str]] = defaultdict(list)
        for name, dims in input_profile.items():
            if len(dims) != 3:
                raise ValueError(f"Expected min/opt/max dimensions for {name}")
            for flag, shape in zip(
                ("--trt-min-shapes", "--trt-opt-shapes", "--trt-max-shapes"),
                dims,
                strict=True,
            ):
                profile_args[flag].append(f"{name}:{str(list(shape)).replace(' ', '')}")
        for flag, shapes in profile_args.items():
            command.append(flag)
            command.extend(shapes)

    print(f"TensorRT {trt_version} strongly typed build command:\n{shlex.join(command)}")
    subprocess.run(command, check=True)


TRTBaseConfig.build_trt_engine = staticmethod(build_trt_engine_trt11)


def engine_device_memory_size_trt11(self: Engine) -> int:
    if Version(trt_version).major < 11:
        return _original_device_memory_size.__get__(self, Engine)
    if self.allocation_policy == "global":
        return self.engine.device_memory_size_v2
    if not self.context.all_binding_shapes_specified:
        return 0
    return self.context.update_device_memory_size_for_shapes()


def engine_cuda_trt11(self: Engine) -> Engine:
    if Version(trt_version).major < 11:
        return _original_engine_cuda(self)
    if self.device.type == "cuda":
        return self
    buffer = self.cpu_engine_buffer if TRT_OFFLOAD_POLICY == "cpu_buffer" else self.engine
    self.engine = engine_from_bytes(buffer)
    gc.collect()
    self.context = self.engine.create_execution_context(trt.ExecutionContextAllocationStrategy.USER_MANAGED)
    memory_size = self.device_memory_size
    self.context_memory.resize(self.__class__.__name__, memory_size)
    self.context.set_device_memory(self.context_memory.shared_device_memory, memory_size)
    return self


def engine_infer_trt11(self: Engine, feed_dict):
    if Version(trt_version).major < 11:
        return _original_engine_infer(self, feed_dict)
    input_hash = self.calculate_input_hash(feed_dict)
    shape_changed = self.current_input_hash != input_hash
    if shape_changed:
        self.override_shapes(feed_dict)
        self.cuda_graph = None
        self.cuda_graph_input_hash = None

    memory_size = self.device_memory_size
    self.context.set_device_memory(self.context_memory.shared_device_memory, memory_size)
    graph_engine_names = {
        name.strip()
        for name in os.getenv("FLUX_TRT_CUDA_GRAPH_ENGINES", "TransformerEngine").split(",")
        if name.strip()
    }
    use_cuda_graph = env_enabled("FLUX_TRT_CUDA_GRAPH") and self.__class__.__name__ in graph_engine_names
    use_dedicated_stream = env_enabled("FLUX_TRT_CUDA_GRAPH") or env_enabled("FLUX_TRT_DEDICATED_STREAM")
    mode = "cuda_graph" if use_cuda_graph else "eager"

    if not use_dedicated_stream:
        for name, tensor in feed_dict.items():
            self.tensors[name].copy_(tensor, non_blocking=True)
        noerror = self.context.execute_async_v3(self.stream.cuda_stream)
        if not noerror:
            raise ValueError("ERROR: inference failed.")
        return self.tensors

    caller_stream = torch.cuda.current_stream()
    engine_stream = self.stream
    engine_stream.wait_stream(caller_stream)
    nvtx_context = (
        torch.cuda.nvtx.range(f"TRT::{self.__class__.__name__}::{mode}")
        if env_enabled("FLUX_TRT_NVTX")
        else nullcontext()
    )
    with nvtx_context:
        with torch.cuda.stream(engine_stream):
            for name, tensor in feed_dict.items():
                self.tensors[name].copy_(tensor, non_blocking=True)

            if not use_cuda_graph:
                noerror = self.context.execute_async_v3(engine_stream.cuda_stream)
                if not noerror:
                    raise ValueError("ERROR: inference failed.")
            elif self.cuda_graph is not None:
                self.cuda_graph.replay()
            else:
                noerror = self.context.execute_async_v3(engine_stream.cuda_stream)
                if not noerror:
                    raise ValueError("ERROR: inference failed.")

        if use_cuda_graph and self.cuda_graph is None:
            # TensorRT requires one enqueue after a shape/profile change before
            # capture. This also warms lazy library initialization. Capture is
            # intentionally completed during the excluded warmup sample.
            engine_stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=engine_stream):
                noerror = self.context.execute_async_v3(engine_stream.cuda_stream)
                if not noerror:
                    raise ValueError("ERROR: CUDA Graph capture enqueue failed.")
            self.cuda_graph = graph
            self.cuda_graph_input_hash = input_hash
            if self not in _captured_engines:
                _captured_engines.append(self)
            with torch.cuda.stream(engine_stream):
                self.cuda_graph.replay()
            print(f"CUDA_GRAPH_CAPTURED engine={self.__class__.__name__} input_hash={input_hash}")

    caller_stream.wait_stream(engine_stream)
    return self.tensors


def init_runtime_with_optional_stream(self: TRTManager):
    _original_init_runtime(self)
    if env_enabled("FLUX_TRT_CUDA_GRAPH") or env_enabled("FLUX_TRT_DEDICATED_STREAM"):
        self.stream = torch.cuda.Stream()


def stop_runtime_with_cuda_graph_cleanup(self: TRTManager):
    # Destroy graph exec objects before their TensorRT contexts/runtime. Letting
    # Python finalize CUDAGraph after TRT teardown can segfault at process exit.
    if _captured_engines:
        torch.cuda.synchronize()
        for engine in _captured_engines:
            engine.cuda_graph = None
            engine.cuda_graph_input_hash = None
        _captured_engines.clear()
        gc.collect()
    _original_stop_runtime(self)


_profile_save_count = 0


def save_image_with_optional_profiler_range(*args, **kwargs):
    """Use the save boundary to capture only post-warmup generation work."""
    global _profile_save_count
    warmup = int(os.getenv("FLUX_NSYS_WARMUP_SAMPLES", "0"))
    measured = int(os.getenv("FLUX_NSYS_MEASURED_SAMPLES", "0"))
    if warmup <= 0 or measured <= 0:
        return _original_save_image(*args, **kwargs)

    _profile_save_count += 1
    final_count = warmup + measured
    if _profile_save_count == final_count:
        torch.cuda.cudart().cudaProfilerStop()
        print(f"NSYS_CAPTURE_STOP samples={measured}")
        return _original_save_image(*args, **kwargs)

    if _profile_save_count <= warmup:
        result = _original_save_image(*args, **kwargs)
        if _profile_save_count == warmup:
            torch.cuda.cudart().cudaProfilerStart()
            print(f"NSYS_CAPTURE_START warmup={warmup} measured={measured}")
        return result

    # Keep image encoding and D2H copies outside the measured CUDA range.
    bound_arguments = inspect.signature(_original_save_image).bind(*args, **kwargs)
    return bound_arguments.arguments["idx"] + 1


Engine.device_memory_size = property(engine_device_memory_size_trt11)
Engine.cuda = engine_cuda_trt11
Engine.infer = engine_infer_trt11
TRTManager.init_runtime = init_runtime_with_optional_stream
TRTManager.stop_runtime = stop_runtime_with_cuda_graph_cleanup
flux_cli.save_image = save_image_with_optional_profiler_range


if __name__ == "__main__":
    Fire(main)
