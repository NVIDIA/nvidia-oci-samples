# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import re
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class RepositoryPortabilityTest(unittest.TestCase):
    def test_no_site_specific_cluster_paths(self):
        banned = (
            "/scratch/fsw/",
            ".sqsh",
            "coreai_devtech_all",
            "customers/oracle",
            "dc2-cdot",
            "gitlab-master.nvidia.com",
            "jhaotingc",
            "nvidia.atlassian.net",
            "oci-aga",
            "squashfs",
            "workspace/2608",
        )
        text_files = []
        for root in (
            "README.md",
            "ENVIRONMENTS.md",
            "INSTALL.md",
            "THIRD_PARTY_NOTICES.md",
            "benchmark.py",
            "benchmarks",
            "configs",
            "docs",
            "reports",
            "scripts",
            "third_party",
            "tools",
        ):
            path = REPOSITORY_ROOT / root
            text_files.extend([path] if path.is_file() else path.rglob("*"))
        for path in text_files:
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            content = path.read_text(encoding="utf-8", errors="ignore").lower()
            for marker in banned:
                self.assertNotIn(marker, content, str(path))

    def test_readme_relative_links_exist(self):
        readme = (REPOSITORY_ROOT / "README.md").read_text(encoding="utf-8")
        links = re.findall(r"\[[^]]+\]\(([^)]+)\)", readme)
        relative_links = [link for link in links if "://" not in link and not link.startswith("#")]
        self.assertTrue(relative_links)
        for link in relative_links:
            path = link.split("#", 1)[0]
            self.assertTrue((REPOSITORY_ROOT / path).exists(), link)

    def test_slurm_example_is_executable(self):
        script = REPOSITORY_ROOT / "scripts/slurm/smoke_hf_diffusers.sbatch"
        self.assertTrue(script.stat().st_mode & 0o111)
        source = script.read_text(encoding="utf-8")
        self.assertIn('"$REPO_ROOT/benchmark.py"', source)


if __name__ == "__main__":
    unittest.main()
