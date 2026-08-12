# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import argparse
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import benchmark
from benchmark import MODES, build_command, missing_paths


def launch_args(**overrides) -> argparse.Namespace:
    values = {
        "mode": "hf-diffusers-bf16-compile",
        "model": None,
        "native_model_dir": None,
        "onnx_dir": None,
        "engine_root": None,
        "batches": [1, 4],
        "warmup": 2,
        "iterations": 20,
        "output_dir": Path("results"),
        "nsys_capture": False,
        "build_only": False,
        "check_only": False,
        "dry_run": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class UnifiedCommandTest(unittest.TestCase):
    def test_exposes_all_supported_modes(self):
        self.assertEqual(len(MODES), 15)
        self.assertIn("trtllm-visualgen-nvfp4-cuda-graph", MODES)
        self.assertIn("torchao-diffusers-nvfp4-regional-cg", MODES)
        self.assertIn("sglang-bf16-offline-compile", MODES)
        self.assertIn("vllm-omni-bf16-offline", MODES)
        for removed_mode in (
            "pytorch-bf16-eager",
            "hf-diffusers-bf16-eager",
            "hf-diffusers-nvfp4-weight-only",
            "hf-diffusers-nvfp4-w4a4",
            "hf-diffusers-nvfp4-w4a4-compile",
            "torchao-diffusers-nvfp4-eager",
        ):
            self.assertNotIn(removed_mode, MODES)

    def test_diffusers_uses_expected_output_layout(self):
        command = build_command(launch_args(model=Path("/models/diffusers")))
        self.assertIn("benchmarks.flux1_schnell.hf_diffusers_flux_sweep", command)
        output_index = command.index("--output-dir") + 1
        self.assertEqual(command[output_index], "results/hf-diffusers")
        self.assertIn("request-batch", command)

    def test_visualgen_selects_cuda_graph_config(self):
        command = build_command(
            launch_args(
                mode="trtllm-visualgen-bf16-cuda-graph",
                model=Path("/models/diffusers"),
            ),
            repository_root=Path("/repo"),
        )
        self.assertEqual(
            command[1:3],
            ["-m", "benchmarks.flux1_schnell.visualgen_flux_sweep"],
        )
        config_index = command.index("--config") + 1
        self.assertEqual(
            command[config_index],
            "/repo/benchmarks/flux1_schnell/configs/visualgen_bf16_cg.yaml",
        )

    def test_pytorch_requires_only_native_checkpoint_argument(self):
        command = build_command(
            launch_args(
                mode="pytorch-bf16-compile",
                native_model_dir=Path("/models/native"),
            )
        )
        self.assertIn("--native-model-dir", command)
        self.assertNotIn("--onnx-dir", command)
        self.assertNotIn("--engine-root", command)

    def test_tensorrt_requires_only_onnx_and_engine_arguments(self):
        command = build_command(
            launch_args(
                mode="trt-bf16-eager",
                onnx_dir=Path("/models/onnx"),
                engine_root=Path("/engines"),
            )
        )
        self.assertIn("--onnx-dir", command)
        self.assertIn("--engine-root", command)
        self.assertNotIn("--native-model-dir", command)

    def test_every_mode_builds_a_python_module_command(self):
        expected_modules = {
            "hf": "benchmarks.flux1_schnell.hf_diffusers_flux_sweep",
            "torchao": "benchmarks.flux1_schnell.torchao_diffusers_flux_sweep",
            "visualgen": "benchmarks.flux1_schnell.visualgen_flux_sweep",
            "pytorch": "benchmarks.flux1_schnell.flux_batch_sweep",
            "trt": "benchmarks.flux1_schnell.flux_batch_sweep",
            "sglang": "benchmarks.flux1_schnell.sglang_flux_sweep",
            "vllm_omni": "benchmarks.flux1_schnell.vllm_omni_flux_sweep",
        }
        for mode, spec in MODES.items():
            with self.subTest(mode=mode):
                command = build_command(
                    launch_args(
                        mode=mode,
                        model=Path("/models/diffusers"),
                        native_model_dir=Path("/models/native"),
                        onnx_dir=Path("/models/onnx"),
                        engine_root=Path("/engines"),
                    )
                )
                self.assertEqual(command[1], "-m")
                self.assertEqual(command[2], expected_modules[spec.runner])
                if spec.runner not in ("torchao", "sglang", "vllm_omni"):
                    self.assertIn("request-batch", command)
                self.assertIn("1024", command)

    def test_offline_backends_use_their_own_runners(self):
        for mode, module in (
            (
                "sglang-bf16-offline-compile",
                "benchmarks.flux1_schnell.sglang_flux_sweep",
            ),
            (
                "vllm-omni-bf16-offline",
                "benchmarks.flux1_schnell.vllm_omni_flux_sweep",
            ),
        ):
            with self.subTest(mode=mode):
                command = build_command(
                    launch_args(mode=mode, model=Path("/models/diffusers"))
                )
                self.assertEqual(command[2], module)
                self.assertIn("--batches", command)
                self.assertIn("4", command)

    def test_offline_backends_support_parent_scoped_capture(self):
        for mode in (
            "sglang-bf16-offline-compile",
            "vllm-omni-bf16-offline",
        ):
            with self.subTest(mode=mode):
                command = build_command(
                    launch_args(
                        mode=mode,
                        model=Path("/models/diffusers"),
                        batches=[1],
                        nsys_capture=True,
                    )
                )
                self.assertIn("--nsys-capture", command)

    def test_capture_execs_runner_in_profiler_target_process(self):
        args = launch_args(
            mode="sglang-bf16-offline-compile",
            model=Path("/models/diffusers"),
            batches=[1],
            nsys_capture=True,
        )
        command = ["/python", "-m", "offline_runner", "--nsys-capture"]
        with (
            mock.patch.object(benchmark, "parse_args", return_value=args),
            mock.patch.object(benchmark, "build_command", return_value=command),
            mock.patch.object(benchmark, "preflight_errors", return_value=[]),
            mock.patch.object(benchmark.os, "chdir"),
            mock.patch.object(
                benchmark.os,
                "execv",
                side_effect=RuntimeError("exec called"),
            ) as execv,
        ):
            with self.assertRaisesRegex(RuntimeError, "exec called"):
                benchmark.main()
        execv.assert_called_once_with(command[0], command)


class PreflightPathTest(unittest.TestCase):
    def test_diffusers_layout_reports_missing_components(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory)
            (model / "model_index.json").touch()
            errors = missing_paths(
                launch_args(model=model), MODES["hf-diffusers-bf16-compile"]
            )
        self.assertTrue(any("transformer" in error for error in errors))
        self.assertTrue(any("tokenizer_2" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
