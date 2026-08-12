# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
NVIDIA_COPYRIGHT = (
    "SPDX-FileCopyrightText: Copyright (c) 2026 "
    "NVIDIA CORPORATION & AFFILIATES. All rights reserved."
)


class RepositoryComplianceTest(unittest.TestCase):
    def test_parent_repository_policy_is_declared(self):
        self.assertFalse((REPOSITORY_ROOT / "LICENSE").exists())
        self.assertFalse((REPOSITORY_ROOT / "CONTRIBUTING.md").exists())
        self.assertTrue((REPOSITORY_ROOT / "THIRD_PARTY_NOTICES.md").is_file())

        readme = (REPOSITORY_ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("NVIDIA/nvidia-oci-samples", readme)
        self.assertIn("Apache License 2.0", readme)
        self.assertIn("CONTRIBUTING.MD", readme)
        self.assertIn("CLA.MD", readme)
        self.assertIn("Sign off every commit", readme)
        self.assertIn(
            "nvidia-oci-samples/generative-ai-samples/flux-1-benchmarking", readme
        )

    def test_text_source_files_have_spdx_headers(self):
        source_files = [REPOSITORY_ROOT / "Makefile"]
        for suffix in ("*.py", "*.yaml", "*.yml", "*.sbatch", "*.sh"):
            source_files.extend(REPOSITORY_ROOT.rglob(suffix))

        for path in source_files:
            if ".git" in path.parts or "__pycache__" in path.parts:
                continue
            header = "\n".join(path.read_text(encoding="utf-8").splitlines()[:15])
            self.assertIn(NVIDIA_COPYRIGHT, header, str(path))
            self.assertIn("SPDX-License-Identifier:", header, str(path))

    def test_adapted_sources_preserve_upstream_licenses(self):
        expected = {
            "benchmarks/flux1_schnell/flux_batch_sweep.py": "Apache-2.0",
            "benchmarks/flux1_schnell/flux_t2i_trt11.py": "Apache-2.0",
            "benchmarks/flux1_schnell/torchao_diffusers_flux_sweep.py": (
                "BSD-3-Clause AND Apache-2.0"
            ),
        }
        for relative_path, expression in expected.items():
            source = (REPOSITORY_ROOT / relative_path).read_text(encoding="utf-8")
            self.assertIn(f"SPDX-License-Identifier: {expression}", source)
            self.assertIn("THIRD_PARTY_NOTICES.md", source)

        notices = (REPOSITORY_ROOT / "THIRD_PARTY_NOTICES.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("802fb4713906133fcbd0d8dc5351620ca4773036", notices)
        self.assertIn("4ee58b57d0fbed12a88abfa86440cd375b9890e1", notices)

    def test_noncommentable_files_have_license_sidecars(self):
        targets = [REPOSITORY_ROOT / "configs/flux1-schnell-1024.json"]
        targets.extend((REPOSITORY_ROOT / "docs/images/nsys").glob("*.png"))
        self.assertGreater(len(targets), 1)
        for target in targets:
            sidecar = Path(f"{target}.license")
            self.assertTrue(sidecar.is_file(), str(sidecar))
            content = sidecar.read_text(encoding="utf-8")
            self.assertIn(NVIDIA_COPYRIGHT, content)
            self.assertIn("SPDX-License-Identifier: Apache-2.0", content)

    def test_redistributable_binary_artifacts_are_not_bundled(self):
        forbidden_suffixes = (
            ".engine",
            ".nsys-rep",
            ".onnx",
            ".plan",
            ".safetensors",
            ".sqsh",
        )
        offenders = [
            path.relative_to(REPOSITORY_ROOT)
            for path in REPOSITORY_ROOT.rglob("*")
            if path.is_file()
            and ".git" not in path.parts
            and path.name.endswith(forbidden_suffixes)
        ]
        self.assertEqual([], offenders)


if __name__ == "__main__":
    unittest.main()
