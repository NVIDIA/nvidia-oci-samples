# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import unittest
from pathlib import Path


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
        self.assertIn('"timing_scope": "offline_api_wall_to_complete_outputs"', source)
        self.assertNotIn("urllib", source)
        self.assertNotIn('"serve"', source)
        self.assertNotIn("sitecustomize", source)
        self.assertNotIn("FLUX_BENCH", source)
        self.assertNotIn("engine-forward", source)
        self.assertIn('"--nsys-capture"', source)
        self.assertIn('range_start("flux_offline_profile")', source)
        self.assertIn("range_end(range_id)", source)

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


if __name__ == "__main__":
    unittest.main()
