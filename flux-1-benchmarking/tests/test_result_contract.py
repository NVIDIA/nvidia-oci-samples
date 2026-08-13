# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import unittest

from benchmarks.flux1_schnell.flux_prompt_bank import prompt_digest, request_prompts
from tools.check_result_contract import contract_errors, output_count


def contract_compliant_result(batch: int = 4) -> dict:
    latency_ms = 1000.0
    return {
        "status": "ok",
        "backend": "test",
        "batch_size": batch,
        "height": 1024,
        "width": 1024,
        "steps": 4,
        "warmup": 2,
        "iterations": 20,
        "batch_semantics": "request-batch",
        "prompt_count": batch,
        "images_per_prompt": 1,
        "prompt_sha256": prompt_digest(request_prompts(batch)),
        "output_count": batch,
        "timing_scope": "host_wall_with_cuda_synchronize",
        "mean_batch_latency_ms": latency_ms,
        "images_per_second": batch / (latency_ms / 1000.0),
    }


class OutputCountTest(unittest.TestCase):
    def test_explicit_output_count(self):
        self.assertEqual(output_count({"output_count": 4}), 4)

    def test_tensor_shape_output_count(self):
        self.assertEqual(output_count({"image_shape": [4, 3, 1024, 1024]}), 4)

    def test_visualgen_output_count(self):
        self.assertEqual(output_count({"image_shapes": [[3, 1024, 1024]] * 4}), 4)

    def test_rejects_malformed_explicit_output_count(self):
        for value in ([], {}, "invalid", True, 4.0):
            with self.subTest(value=value):
                self.assertIsNone(output_count({"output_count": value}))

    def test_rejects_malformed_shape_count(self):
        for value in ([], {}, "invalid", ["4", 3, 1024, 1024], [True]):
            with self.subTest(value=value):
                self.assertIsNone(output_count({"image_shape": value}))

    def test_rejects_malformed_image_shapes(self):
        for value in (
            "invalid",
            [None],
            [["bad"]],
            [[3, 1024, 1024], None],
            [[True, 1024, 1024]],
            [[3.0, 1024, 1024]],
            [[3, 0, 1024]],
        ):
            with self.subTest(value=value):
                self.assertIsNone(output_count({"image_shapes": value}))


class ResultContractTest(unittest.TestCase):
    def test_contract_compliant_result(self):
        self.assertEqual(
            contract_errors(
                contract_compliant_result(),
                expected_warmup=2,
                expected_iterations=20,
            ),
            [],
        )

    def test_rejects_repeated_prompt_semantics(self):
        result = contract_compliant_result()
        result["batch_semantics"] = "images-per-prompt"
        errors = contract_errors(result)
        self.assertTrue(any("request-batch" in error for error in errors))

    def test_rejects_wrong_output_count(self):
        result = contract_compliant_result()
        result["output_count"] = 1
        errors = contract_errors(result)
        self.assertTrue(any("output_count" in error for error in errors))

    def test_rejects_malformed_image_shapes(self):
        result = contract_compliant_result()
        result.pop("output_count")
        result["image_shapes"] = [[3, 1024, 1024], None, None, None]
        errors = contract_errors(result)
        self.assertTrue(any("output_count" in error for error in errors))

    def test_rejects_inconsistent_throughput(self):
        result = contract_compliant_result()
        result["images_per_second"] = 99.0
        errors = contract_errors(result)
        self.assertTrue(any("inconsistent" in error for error in errors))

    def test_rejects_incomplete_engine_api(self):
        result = contract_compliant_result()
        for key in (
            "client_completed_after_engine_forward",
            "api_completed_after_engine_forward",
            "api_completed_with_all_outputs",
        ):
            with self.subTest(key=key):
                result[key] = False
                errors = contract_errors(result)
                self.assertTrue(any("engine API" in error for error in errors))
                result.pop(key)

    def test_rejects_wrong_realized_request_batch(self):
        result = contract_compliant_result()
        result["realized_request_batch_sizes"] = [4] * 19 + [2]
        errors = contract_errors(result)
        self.assertTrue(any("realized request batch" in error for error in errors))

    def test_rejects_malformed_realized_request_batches(self):
        result = contract_compliant_result()
        result["realized_request_batch_sizes"] = 4
        errors = contract_errors(result)
        self.assertTrue(any("realized request batch" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
