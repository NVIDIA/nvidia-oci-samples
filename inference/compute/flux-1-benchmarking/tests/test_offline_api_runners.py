# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from benchmarks.flux1_schnell import vllm_omni_flux_sweep


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class OfflineApiRunnerTest(unittest.TestCase):
    def runner_source(self, name: str) -> str:
        return (
            REPOSITORY_ROOT / "benchmarks" / "flux1_schnell" / name
        ).read_text(encoding="utf-8")

    def test_sglang_uses_local_diffgenerator_without_http(self):
        source = self.runner_source("sglang_flux_sweep.py")
        self.assertIn("DiffGenerator.from_pretrained", source)
        self.assertIn("local_mode=True", source)
        self.assertIn("async_scheduler_client", source)
        self.assertIn('"--batching-delay-ms", type=float, default=100.0', source)
        self.assertIn('"--request-timeout-seconds", type=float, default=600.0', source)
        self.assertIn("await asyncio.wait_for", source)
        self.assertIn('"timing_scope": "offline_api_wall_to_complete_outputs"', source)
        self.assertNotIn("urllib", source)
        self.assertNotIn('"serve"', source)
        self.assertNotIn("sitecustomize", source)
        self.assertNotIn("FLUX_BENCH", source)
        self.assertNotIn("engine-forward", source)
        self.assertIn('"--nsys-capture"', source)
        self.assertIn('range_start("flux_offline_profile")', source)
        self.assertIn("range_end(range_id)", source)
        self.assertIn('"error_type": type(exc).__name__', source)
        self.assertIn('run_dir / "load.json"', source)

    def test_vllm_omni_uses_offline_omni_without_http(self):
        source = self.runner_source("vllm_omni_flux_sweep.py")
        self.assertIn("from vllm_omni.entrypoints.omni import Omni", source)
        self.assertIn("omni.generate", source)
        self.assertIn('"--request-batch-max-wait-ms", type=float, default=100.0', source)
        self.assertIn(
            "request_batch_max_wait_ms=args.request_batch_max_wait_ms", source
        )
        self.assertIn('"timing_scope": "offline_api_wall_to_complete_outputs"', source)
        self.assertNotIn("urllib", source)
        self.assertNotIn('"serve"', source)
        self.assertNotIn("sitecustomize", source)
        self.assertNotIn("FLUX_BENCH", source)
        self.assertNotIn("engine-forward", source)
        self.assertIn('"--nsys-capture"', source)
        self.assertIn('range_start("flux_offline_profile")', source)
        self.assertIn("range_end(range_id)", source)
        self.assertIn('"error_type": type(exc).__name__', source)
        self.assertIn('run_dir / "load.json"', source)
        self.assertIn("finally:\n        omni.close()", source)

    def test_vllm_omni_rejects_nonpositive_workloads_before_import(self):
        cases = (
            (["--batches", "0"], "--batches values must be positive"),
            (["--batches", "-1"], "--batches values must be positive"),
            (
                ["--batches", "1", "--iterations", "0"],
                "--iterations must be positive",
            ),
            (
                ["--batches", "1", "--iterations", "-1"],
                "--iterations must be positive",
            ),
            (
                ["--batches", "1", "--warmup", "-1"],
                "--warmup must be non-negative",
            ),
        )
        for workload_args, expected_error in cases:
            with self.subTest(workload_args=workload_args):
                stderr = io.StringIO()
                argv = [
                    "vllm_omni_flux_sweep.py",
                    "--model",
                    "/models/flux",
                    *workload_args,
                    "--output-dir",
                    "/results",
                ]
                with (
                    mock.patch.object(sys, "argv", argv),
                    mock.patch.object(sys, "stderr", stderr),
                    self.assertRaises(SystemExit) as raised,
                ):
                    vllm_omni_flux_sweep.main()
                self.assertEqual(raised.exception.code, 2)
                self.assertIn(expected_error, stderr.getvalue())

    def test_visualgen_rejects_nonpositive_workloads_before_import(self):
        with mock.patch.dict(sys.modules, {"torch": types.ModuleType("torch")}):
            from benchmarks.flux1_schnell import visualgen_flux_sweep

        cases = (
            (["--batches", "0"], "--batches values must be positive"),
            (["--batches", "-1"], "--batches values must be positive"),
            (
                ["--batches", "1", "--iterations", "0"],
                "--iterations must be positive",
            ),
            (
                ["--batches", "1", "--iterations", "-1"],
                "--iterations must be positive",
            ),
            (
                ["--batches", "1", "--warmup", "-1"],
                "--warmup must be non-negative",
            ),
        )
        for workload_args, expected_error in cases:
            with self.subTest(workload_args=workload_args):
                stderr = io.StringIO()
                argv = [
                    "visualgen_flux_sweep.py",
                    "--model",
                    "/models/flux",
                    "--config",
                    "/configs/visualgen.yaml",
                    "--precision",
                    "bf16",
                    *workload_args,
                    "--output-dir",
                    "/results",
                ]
                with (
                    mock.patch.object(sys, "argv", argv),
                    mock.patch.object(sys, "stderr", stderr),
                    self.assertRaises(SystemExit) as raised,
                ):
                    visualgen_flux_sweep.main()
                self.assertEqual(raised.exception.code, 2)
                self.assertIn(expected_error, stderr.getvalue())

    def test_vllm_omni_records_lookup_failure_and_closes_engine(self):
        class FakeOmni:
            instance = None

            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.closed = False
                FakeOmni.instance = self

            def close(self):
                self.closed = True

        vllm_package = types.ModuleType("vllm_omni")
        entrypoints_package = types.ModuleType("vllm_omni.entrypoints")
        omni_module = types.ModuleType("vllm_omni.entrypoints.omni")
        extras_module = types.ModuleType("vllm_omni.model_extras")
        omni_module.Omni = FakeOmni
        extras_module.get_model_class_name = mock.Mock(
            side_effect=RuntimeError("model class lookup failed")
        )
        fake_modules = {
            "vllm_omni": vllm_package,
            "vllm_omni.entrypoints": entrypoints_package,
            "vllm_omni.entrypoints.omni": omni_module,
            "vllm_omni.model_extras": extras_module,
        }

        with tempfile.TemporaryDirectory() as directory:
            argv = [
                "vllm_omni_flux_sweep.py",
                "--model",
                "/models/flux",
                "--batches",
                "1",
                "--output-dir",
                directory,
            ]
            with (
                mock.patch.dict(sys.modules, fake_modules),
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(sys, "stdout", io.StringIO()),
                mock.patch.object(
                    vllm_omni_flux_sweep.importlib.metadata,
                    "version",
                    return_value="test",
                ),
                self.assertRaisesRegex(RuntimeError, "model class lookup failed"),
            ):
                vllm_omni_flux_sweep.main()

            self.assertIsNotNone(FakeOmni.instance)
            self.assertTrue(FakeOmni.instance.closed)
            load_path = Path(directory) / vllm_omni_flux_sweep.MODE_DIR / "load.json"
            load_result = json.loads(load_path.read_text(encoding="utf-8"))
            self.assertEqual(load_result["status"], "error")
            self.assertEqual(load_result["error_type"], "RuntimeError")
            self.assertEqual(load_result["error"], "model class lookup failed")
            self.assertIn(
                "RuntimeError: model class lookup failed", load_result["traceback"]
            )


if __name__ == "__main__":
    unittest.main()
