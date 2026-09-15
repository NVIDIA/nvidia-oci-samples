# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests that run without a GPU or a model.

These cover the parts that are easy to get wrong and expensive to discover on a
node: stage selection, workspace resolution, manifest round-tripping, config
validity, and the manifest validator.
"""

from __future__ import annotations

import json
import sys
import tempfile
import os
import unittest
from unittest import mock
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY_ROOT))

import quantize  # noqa: E402
from common import paths  # noqa: E402
from common.manifest import Manifest  # noqa: E402
from tools import check_run_manifest  # noqa: E402
from stages import s03_static_export  # noqa: E402
from stages import s05_schema_diff  # noqa: E402


class TestStageSelection(unittest.TestCase):
    def _args(self, **overrides):
        parser = quantize.build_parser()
        args = parser.parse_args([])
        for key, value in overrides.items():
            setattr(args, key, value)
        return args

    def test_default_runs_every_stage_in_order(self):
        selected = quantize.select(self._args())
        self.assertEqual([s.name for s in selected], [s.name for s in quantize.STAGES])

    def test_from_and_through_slice_inclusively(self):
        selected = quantize.select(self._args(from_stage="download", through="export"))
        self.assertEqual([s.name for s in selected], ["download", "dynamic", "export"])

    def test_explicit_stages_keep_pipeline_order(self):
        selected = quantize.select(self._args(stage=["export", "preflight"]))
        self.assertEqual([s.name for s in selected], ["preflight", "export"])

    def test_unknown_stage_is_rejected(self):
        with self.assertRaises(SystemExit):
            quantize.select(self._args(stage=["nope"]))

    def test_reversed_range_is_rejected(self):
        with self.assertRaises(SystemExit):
            quantize.select(self._args(from_stage="verify", through="download"))

    def test_every_stage_module_is_importable_and_has_run(self):
        import importlib

        for stage in quantize.STAGES:
            module = importlib.import_module(stage.module)
            self.assertTrue(callable(getattr(module, "run", None)), f"{stage.name} has no run()")


class TestWorkspace(unittest.TestCase):
    def test_explicit_path_is_used_and_directories_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = paths.resolve(tmp)
            workspace.ensure()
            self.assertEqual(workspace.source, "--workspace")
            for directory in workspace.directories():
                self.assertTrue(directory.is_dir())

    def test_node_local_paths_are_flagged_as_ephemeral(self):
        self.assertTrue(paths._looks_ephemeral(Path("/raid/someone")))
        self.assertFalse(paths._looks_ephemeral(Path("/home/scratch.example_team")))

    def test_ephemeral_workspace_produces_a_warning(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = paths.Workspace(root=Path(tmp), source="test", ephemeral=True, free_gb=500)
            self.assertTrue(any("node-local" in w for w in workspace.warnings()))

    def test_low_disk_produces_a_warning(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = paths.Workspace(root=Path(tmp), source="test", ephemeral=False, free_gb=5)
            self.assertTrue(any("free" in w for w in workspace.warnings()))


class TestSharedModels(unittest.TestCase):
    """Checkpoints are shared across a team; outputs are not."""

    def setUp(self):
        # paths.resolve() consults the real environment, and anyone who has
        # actually run this harness has FLUX_QUANT_MODELS exported -- INSTALL.md
        # tells them to. That made these tests pass on a clean machine and fail
        # on a working one, which is the wrong way round.
        patcher = mock.patch.dict(os.environ, {}, clear=False)
        patcher.start()
        self.addCleanup(patcher.stop)
        for name in ("FLUX_QUANT_MODELS", "FLUX_QUANT_WORKSPACE", "FLUX_QUANT_SHARED_CACHES"):
            os.environ.pop(name, None)

    @staticmethod
    def _tmp_root(tmp: str) -> Path:
        """Resolve the temporary directory before building paths from it.

        ``_shared_models_root`` resolves internally, and it has to: on the
        cluster a workspace is frequently reached through a symlinked scratch
        volume, and the parent walk only finds ``models/`` on the real path. So
        it returns a resolved path, and any assertion comparing against one
        built here must start from a resolved path too.

        On macOS this is the difference between passing and failing. ``tempfile``
        hands back ``/var/folders/...`` and ``/var`` is a symlink to
        ``/private/var``, so the resolved and unresolved forms differ. Linux has
        no such symlink, which is why this only ever failed on a laptop.
        """
        return Path(tmp).resolve()

    def test_models_default_to_the_workspace(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._tmp_root(tmp)
            workspace = paths.resolve(str(root))
            self.assertEqual(workspace.models, root / "models")
            self.assertEqual(workspace.models_source, "workspace")

    def test_explicit_models_dir_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._tmp_root(tmp)
            shared = root / "elsewhere"
            workspace = paths.resolve(str(root), models_dir=str(shared))
            self.assertEqual(workspace.models, shared)
            self.assertEqual(workspace.models_source, "--models-dir")

    def test_shared_volume_models_are_found_when_they_exist(self):
        # Mirrors the real layout: /home/scratch.<team>/{models,<user>/flux-quant}
        with tempfile.TemporaryDirectory() as tmp:
            volume = self._tmp_root(tmp) / "scratch.example_team"
            shared = volume / "models"
            shared.mkdir(parents=True)
            workspace_root = volume / "someone" / "flux-quant"
            workspace_root.mkdir(parents=True)

            self.assertEqual(paths._shared_models_root(workspace_root), shared)

    def test_shared_volume_is_ignored_until_the_directory_exists(self):
        # Opting in is a single mkdir; without it we must not invent a path.
        with tempfile.TemporaryDirectory() as tmp:
            volume = self._tmp_root(tmp) / "scratch.example_team"
            workspace_root = volume / "someone" / "flux-quant"
            workspace_root.mkdir(parents=True)

            self.assertIsNone(paths._shared_models_root(workspace_root))

    def test_non_scratch_paths_do_not_get_a_shared_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._tmp_root(tmp) / "plain" / "workspace"
            root.mkdir(parents=True)
            self.assertIsNone(paths._shared_models_root(root))

    def test_shared_volume_is_found_through_a_symlink(self):
        """The macOS failure, reproduced deliberately so it cannot come back.

        A symlinked path to a scratch volume is the normal case on the cluster,
        not an edge case, and it is the exact shape that broke this suite on a
        laptop while passing in CI.
        """
        with tempfile.TemporaryDirectory() as tmp:
            real = self._tmp_root(tmp) / "real"
            volume = real / "scratch.example_team"
            shared = volume / "models"
            shared.mkdir(parents=True)
            (volume / "someone" / "flux-quant").mkdir(parents=True)

            link = self._tmp_root(tmp) / "link"
            link.symlink_to(real, target_is_directory=True)

            via_link = link / "scratch.example_team" / "someone" / "flux-quant"
            self.assertEqual(paths._shared_models_root(via_link), shared)

    def test_ensure_creates_the_shared_models_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._tmp_root(tmp)
            shared = root / "shared-models"
            workspace = paths.resolve(str(root / "ws"), models_dir=str(shared))
            workspace.ensure()
            self.assertTrue(shared.is_dir())
            self.assertTrue(workspace.exports.is_dir())


class TestManifest(unittest.TestCase):
    def test_records_persist_and_reload(self):
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp)
            manifest = Manifest(results)
            manifest.record("preflight", status="ok", outputs={"free_gb": 400})
            self.assertEqual(Manifest(results).stage_status("preflight"), "ok")

    def test_latest_status_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Manifest(Path(tmp))
            manifest.record("export", status="failed")
            manifest.record("export", status="ok")
            self.assertEqual(manifest.stage_status("export"), "ok")

    def test_corrupt_manifest_is_preserved_not_lost(self):
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp)
            (results / "run_manifest.json").write_text("{ not json")
            Manifest(results)
            self.assertTrue((results / "run_manifest.json.corrupt").exists())

    def test_non_dict_environment_is_replaced_not_preserved(self):
        """Keeping a string here breaks a resumed run several stages later."""
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp)
            (results / "run_manifest.json").write_text(
                json.dumps({"environment": "broken", "stages": []})
            )
            manifest = Manifest(results)
            self.assertIsInstance(manifest.data["environment"], dict)
            self.assertIn("hostname", manifest.data["environment"])

    def test_malformed_stage_entries_are_dropped_not_raised_on(self):
        """A partly readable manifest is still worth resuming from."""
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp)
            (results / "run_manifest.json").write_text(
                json.dumps(
                    {
                        "environment": {},
                        "stages": [
                            "not-a-dict",
                            {"no": "stage key"},
                            {"stage": "export", "status": "ok"},
                        ],
                    }
                )
            )
            manifest = Manifest(results)
            self.assertEqual(len(manifest.data["stages"]), 1)
            # The survivor is still usable, and recording onto it still works.
            self.assertEqual(manifest.stage_status("export"), "ok")
            manifest.record("quality", status="ok")
            self.assertEqual(Manifest(results).stage_status("quality"), "ok")

    def test_stage_entries_with_wrongly_typed_fields_are_dropped(self):
        """Presence is not enough: the field types are what later code relies on.

        ``stage`` becomes a dictionary key in the validator, so a list there is
        unhashable; ``status`` is read positionally by ``stage_status``, so an
        entry naming a stage and carrying no status raises KeyError on resume.
        Both survived a presence-only filter.
        """
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp)
            (results / "run_manifest.json").write_text(
                json.dumps(
                    {
                        "environment": {},
                        "stages": [
                            {"stage": [], "status": "ok"},
                            {"stage": "export"},
                            {"stage": "verify", "status": 0},
                            {"stage": "quality", "status": "ok"},
                        ],
                    }
                )
            )
            manifest = Manifest(results)
            self.assertEqual(
                [entry["stage"] for entry in manifest.data["stages"]], ["quality"]
            )
            # The one well-formed entry is untouched, and the dropped ones do
            # not resurface as an exception when they are read back.
            self.assertEqual(manifest.stage_status("quality"), "ok")
            self.assertIsNone(manifest.stage_status("export"))


class TestConfigs(unittest.TestCase):
    def _configs(self):
        return sorted((REPOSITORY_ROOT / "configs").glob("*.json"))

    def test_configs_are_valid_json_with_required_keys(self):
        required = ("modelopt_model", "baseline_repo", "baseline_dir", "generation", "quantization")
        for path in self._configs():
            if path.name == "prompts.json":
                continue
            config = json.loads(path.read_text())
            for key in required:
                self.assertIn(key, config, f"{path.name} missing {key}")

    def test_generation_settings_are_paired_comparison_safe(self):
        for path in self._configs():
            if path.name == "prompts.json":
                continue
            generation = json.loads(path.read_text())["generation"]
            self.assertGreaterEqual(len(generation["seeds"]), 2, f"{path.name} needs at least two seeds")
            self.assertEqual(generation["height"] % 16, 0)
            self.assertEqual(generation["width"] % 16, 0)

    def test_prompt_set_covers_the_named_failure_modes(self):
        prompts = json.loads((REPOSITORY_ROOT / "configs" / "prompts.json").read_text())["prompts"]
        categories = {p["category"] for p in prompts}
        # Text alignment and deformation are the usual fail modes for this workload.
        for expected in ("text-rendering", "counting", "spatial", "anatomy"):
            self.assertIn(expected, categories)
        self.assertEqual(len({p["id"] for p in prompts}), len(prompts), "duplicate prompt ids")

    def test_every_config_has_a_licence_sidecar(self):
        for path in self._configs():
            self.assertTrue(path.with_suffix(".json.license").exists(), f"{path.name} has no .license")


class TestManifestValidator(unittest.TestCase):
    def _write(self, tmp: Path, payload: dict) -> Path:
        path = tmp / "run_manifest.json"
        path.write_text(json.dumps(payload))
        return path

    def test_missing_manifest_is_an_error(self):
        errors, _ = check_run_manifest.check(Path("/nonexistent/run_manifest.json"))
        self.assertTrue(errors)

    def test_missing_gpu_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(
                Path(tmp),
                {
                    "environment": {
                        "hostname": "n",
                        "cpu_arch": "x86_64",
                        "python": "3.12",
                        "gpu": {"available": False},
                        "packages": {"torch": "2.9", "diffusers": "0.36"},
                    },
                    "stages": [],
                },
            )
            errors, _ = check_run_manifest.check(path)
            self.assertTrue(any("No GPU" in e for e in errors))

    def test_non_blackwell_gpu_warns(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(
                Path(tmp),
                {
                    "environment": {
                        "hostname": "n",
                        "cpu_arch": "x86_64",
                        "python": "3.12",
                        "gpu": {"available": True, "architecture": ["9.0"]},
                        "packages": {"torch": "2.9", "diffusers": "0.36"},
                    },
                    "stages": [{"stage": "preflight", "status": "ok", "outputs": {}}],
                },
            )
            _, warnings = check_run_manifest.check(path)
            self.assertTrue(any("Blackwell" in w for w in warnings))

    def test_failed_stage_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(
                Path(tmp),
                {
                    "environment": {
                        "hostname": "n",
                        "cpu_arch": "x86_64",
                        "python": "3.12",
                        "gpu": {"available": True, "architecture": ["10.0"]},
                        "packages": {"torch": "2.9", "diffusers": "0.36"},
                    },
                    "stages": [{"stage": "export", "status": "failed", "outputs": {}}],
                },
            )
            errors, _ = check_run_manifest.check(path)
            self.assertTrue(any("export" in e for e in errors))

    # A validator must report on a malformed manifest, never crash on one. Each
    # of these raised before the type guards went in.

    def test_string_environment_is_reported_not_raised(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(Path(tmp), {"environment": "broken", "stages": []})
            errors, _ = check_run_manifest.check(path)
            self.assertTrue(any("environment is not an object" in e for e in errors))

    def test_string_gpu_and_packages_are_reported_not_raised(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(
                Path(tmp),
                {
                    "environment": {
                        "hostname": "n",
                        "cpu_arch": "x86_64",
                        "python": "3.12",
                        "gpu": "nope",
                        "packages": "nope",
                    },
                    "stages": [],
                },
            )
            errors, _ = check_run_manifest.check(path)
            self.assertTrue(any("environment.gpu is not an object" in e for e in errors))
            self.assertTrue(any("environment.packages is not an object" in e for e in errors))

    def test_non_dict_stage_entry_is_reported_not_raised(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(Path(tmp), {"environment": {}, "stages": ["not-a-dict"]})
            errors, _ = check_run_manifest.check(path)
            self.assertTrue(any("stages[0] is not an object" in e for e in errors))

    def test_stage_entry_without_a_stage_field_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(Path(tmp), {"environment": {}, "stages": [{"status": "ok"}]})
            errors, _ = check_run_manifest.check(path)
            self.assertTrue(any("has no 'stage' field" in e for e in errors))

    def test_stage_entry_without_a_status_is_not_read_as_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(Path(tmp), {"environment": {}, "stages": [{"stage": "export"}]})
            errors, _ = check_run_manifest.check(path)
            self.assertTrue(any("export" in e and "None" in e for e in errors))

    def test_stages_not_a_list_is_reported_not_raised(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(Path(tmp), {"environment": {}, "stages": "nope"})
            errors, _ = check_run_manifest.check(path)
            self.assertTrue(any("stages is not a list" in e for e in errors))

    def test_every_check_still_runs_after_a_bad_shape(self):
        """The point of returning errors rather than raising: one run, every problem."""
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(Path(tmp), {"environment": "broken", "stages": ["not-a-dict", {}]})
            errors, warnings = check_run_manifest.check(path)
            self.assertTrue(any("environment is not an object" in e for e in errors))
            self.assertTrue(any("No GPU" in e for e in errors))
            self.assertTrue(any("No version recorded for torch" in e for e in errors))
            self.assertTrue(any("stages[0] is not an object" in e for e in errors))
            self.assertTrue(any("stages[1] has no 'stage' field" in e for e in errors))
            self.assertTrue(any("never run" in w for w in warnings))

    def test_non_string_stage_name_is_reported_not_raised(self):
        """The name is used as a dictionary key, so an unhashable one stops the run."""
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(
                Path(tmp), {"environment": "broken", "stages": [{"stage": [], "status": "ok"}]}
            )
            errors, _ = check_run_manifest.check(path)
            self.assertTrue(any("stages[0].stage is not a string" in e for e in errors))
            # The checks after the stage loop still ran.
            self.assertTrue(any("No GPU" in e for e in errors))

    def test_downloads_not_a_list_is_reported_not_raised(self):
        """A scalar here was iterated directly, which ended the validation."""
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(
                Path(tmp),
                {
                    "environment": "broken",
                    "stages": [{"stage": "download", "status": "ok", "outputs": {"downloads": 1}}],
                },
            )
            errors, _ = check_run_manifest.check(path)
            self.assertTrue(any("download.downloads is not a list" in e for e in errors))
            self.assertTrue(any("No GPU" in e for e in errors))


class TestSchemaClassification(unittest.TestCase):
    """The exclusion check is what an adopter would rely on, so it is tested directly.

    The trap being guarded against: FLUX has a model-level `proj_out` and a
    `proj_out` inside every single transformer block. A leaf-name match cannot
    tell them apart, and calling the per-block projections wrongly-quantized
    would be a false alarm on 38 layers.
    """

    def _tensors(self):
        """A miniature FLUX-shaped export: two blocks plus top-level modules."""
        records = {}

        def add(name, dtype):
            records[f"transformer/{name}"] = {
                "component": "transformer",
                "name": name,
                "shape": [16, 16],
                "dtype": dtype,
                "file": "model.safetensors",
            }

        # Top-level modules the filter protects: still BF16.
        add("proj_out.weight", "BF16")
        add("x_embedder.weight", "BF16")
        add("context_embedder.weight", "BF16")
        # A layer nested inside a protected module: also meant to stay high.
        add("time_text_embed.timestep_embedder.linear_1.weight", "BF16")
        # Per-block projections sharing the leaf name: legitimately quantized.
        for index in range(2):
            add(f"single_transformer_blocks.{index}.proj_out.weight", "U8")
            add(f"single_transformer_blocks.{index}.proj_out.weight_scale", "F8_E4M3")
            add(f"single_transformer_blocks.{index}.attn.to_q.weight", "U8")
        return records

    def test_dtype_decides_state_not_name(self):
        states = s05_schema_diff.classify_modules(self._tensors(), "transformer")
        self.assertEqual(states["proj_out"], "high-precision")
        self.assertEqual(states["single_transformer_blocks.0.proj_out"], "quantized")
        self.assertEqual(states["single_transformer_blocks.1.attn.to_q"], "quantized")

    def test_scale_tensors_do_not_create_modules(self):
        states = s05_schema_diff.classify_modules(self._tensors(), "transformer")
        self.assertNotIn("single_transformer_blocks.0.proj_out.weight", states)

    def test_top_level_and_nested_never_merge(self):
        states = s05_schema_diff.classify_modules(self._tensors(), "transformer")
        report = s05_schema_diff.exclusion_report(
            states, s05_schema_diff.DOCUMENTED_EXCLUSIONS
        )
        # The protected top-level layer is reported on its own, at high precision.
        self.assertEqual(report["top_level"]["proj_out"], "high-precision")
        # Anything beneath a protected module counts as protected too, even
        # though its own leaf name is not on the exclusion list.
        self.assertEqual(
            report["top_level"]["time_text_embed.timestep_embedder.linear_1"],
            "high-precision",
        )
        # The two per-block projections collapse into one row, and are not
        # mistaken for the protected layer.
        self.assertEqual(
            report["nested"]["single_transformer_blocks.N.proj_out"], {"quantized": 2}
        )
        self.assertNotIn("single_transformer_blocks.N.proj_out", report["top_level"])

    def test_filter_disagreement_is_reported(self):
        states = s05_schema_diff.classify_modules(self._tensors(), "transformer")

        def leaky_filter(module: str) -> bool:
            """Claims to protect every proj_out, including per-block ones."""
            return module.split(".")[-1] in ("proj_out", "x_embedder", "context_embedder")

        agreement = s05_schema_diff.check_filter_agreement(states, leaky_filter)
        self.assertFalse(agreement["agrees"])
        self.assertIn(
            "single_transformer_blocks.0.proj_out", agreement["excluded_but_quantized"]
        )

    def test_filter_agreement_when_recipe_is_correct(self):
        states = s05_schema_diff.classify_modules(self._tensors(), "transformer")

        def exact_filter(module: str) -> bool:
            """Protects only the top-level modules, as ModelOpt intends."""
            return module in ("proj_out", "x_embedder", "context_embedder")

        agreement = s05_schema_diff.check_filter_agreement(states, exact_filter)
        self.assertTrue(agreement["agrees"])
        self.assertEqual(agreement["excluded_but_quantized"], [])

    def test_coverage_counts_modules_not_tensors(self):
        states = s05_schema_diff.classify_modules(self._tensors(), "transformer")
        coverage = s05_schema_diff._coverage(states)
        # 4 protected at BF16: proj_out, x_embedder, context_embedder, and the
        # linear nested under time_text_embed.
        # 4 quantized: proj_out and attn.to_q in each of the 2 blocks.
        # The weight_scale tensors are metadata and must not count as modules.
        self.assertEqual(coverage["high_precision"], 4)
        self.assertEqual(coverage["quantized"], 4)
        self.assertEqual(coverage["weight_bearing_modules"], 8)

    def test_shapes_match_across_naming_conventions(self):
        """Same shapes means servable; tensor names are irrelevant.

        The published checkpoint is ComfyUI format and ours is Diffusers, so
        tensor names share nothing. Matching on (dtype, shape) is what makes the
        comparison possible at all.
        """
        ours = {
            "transformer/single_transformer_blocks.0.proj_out.weight": {
                "component": "transformer", "name": "x", "shape": [64, 32], "dtype": "U8",
            },
            "transformer/single_transformer_blocks.1.proj_out.weight": {
                "component": "transformer", "name": "y", "shape": [64, 32], "dtype": "U8",
            },
        }
        theirs = {
            "root/double_blocks.0.img_mlp.0.weight": {
                "component": "root", "name": "a", "shape": [64, 32], "dtype": "U8",
            },
            "root/double_blocks.1.img_mlp.0.weight": {
                "component": "root", "name": "b", "shape": [64, 32], "dtype": "U8",
            },
        }
        result = s05_schema_diff.compare_shapes(ours, theirs, "transformer", "root")
        self.assertEqual(result["tensors_matched_by_signature"], 2)
        self.assertEqual(result["share_matched"], 1.0)
        self.assertEqual(result["only_in_ours"], [])

    def test_shape_mismatch_is_surfaced_not_averaged(self):
        ours = {
            "t/a": {"component": "t", "name": "a", "shape": [64, 32], "dtype": "U8"},
            "t/b": {"component": "t", "name": "b", "shape": [128, 128], "dtype": "BF16"},
        }
        theirs = {
            "t/c": {"component": "t", "name": "c", "shape": [64, 32], "dtype": "U8"},
        }
        result = s05_schema_diff.compare_shapes(ours, theirs, "t", "t")
        self.assertEqual(result["tensors_matched_by_signature"], 1)
        self.assertEqual(len(result["only_in_ours"]), 1)
        self.assertEqual(result["only_in_ours"][0]["shape"], [128, 128])

    def test_inventory_lists_every_tensor(self):
        import csv as _csv

        tensors = {
            "transformer/a": {
                "component": "transformer", "name": "a.weight", "shape": [4, 8], "dtype": "U8",
            },
            "vae/b": {
                "component": "vae", "name": "b.weight", "shape": [2], "dtype": "BF16",
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "inv.csv"
            count = s05_schema_diff.write_inventory(tensors, path)
            self.assertEqual(count, 2)
            rows = list(_csv.DictReader(path.open()))
            self.assertEqual(len(rows), 2)
            by_name = {r["tensor"]: r for r in rows}
            self.assertEqual(by_name["a.weight"]["shape"], "4x8")
            self.assertEqual(by_name["a.weight"]["elements"], "32")
            self.assertEqual(by_name["b.weight"]["dtype"], "BF16")

    def test_fp8_weights_are_not_confused_with_block_scales(self):
        """The distinction the whole recipe comparison turns on.

        An NVFP4 weight is a U8 tensor plus an F8_E4M3 block scale, so a naive
        dtype histogram shows F8 entries that are metadata. An F8 tensor that is
        a *weight* means that layer was quantized to 8 bits instead of 4 — a
        different recipe, not a different file format.
        """
        tensors = {
            "t/a": {  # NVFP4 weight
                "component": "t", "name": "blk.0.attn.weight", "shape": [3072, 1536], "dtype": "U8",
            },
            "t/b": {  # its block scale, must not count as a weight
                "component": "t", "name": "blk.0.attn.weight_scale",
                "shape": [3072, 192], "dtype": "F8_E4M3",
            },
            "t/c": {  # a genuine FP8 weight
                "component": "t", "name": "blk.0.norm1.linear.weight",
                "shape": [18432, 3072], "dtype": "F8_E4M3",
            },
            "t/d": {
                "component": "t", "name": "blk.0.norm_out.weight", "shape": [64], "dtype": "BF16",
            },
        }
        roles = s05_schema_diff.precision_by_role(tensors, "t")
        self.assertEqual(roles["weight_tensors"], 3)
        self.assertEqual(roles["by_precision"]["4-bit (packed)"], 1)
        self.assertEqual(roles["by_precision"]["8-bit float"], 1)
        self.assertEqual(roles["by_precision"]["high precision (BF16)"], 1)
        # The block scale is metadata and appears nowhere in the weight counts.
        self.assertNotIn("blk.0.attn.weight_scale", str(roles["examples"]))

    def test_rank_explains_unquantized_layers(self):
        """A 1-D weight is a norm vector, not a GEMM. FLUX has 152 of them.

        Reporting those as "the filter permitted this but it wasn't quantized"
        without the reason invites a false alarm; a 2-D weight in the same
        bucket is genuinely worth investigating.
        """
        tensors = {
            "t/a": {  # QK-norm: 1-D, nothing to accelerate
                "component": "t", "name": "blk.0.attn.norm_q.weight",
                "shape": [128], "dtype": "BF16",
            },
            "t/b": {
                "component": "t", "name": "blk.0.attn.norm_k.weight",
                "shape": [128], "dtype": "BF16",
            },
            "t/c": {  # a real Linear left alone: worth a look
                "component": "t", "name": "blk.0.ff.net.2.weight",
                "shape": [3072, 3072], "dtype": "BF16",
            },
        }
        modules = [
            "blk.0.attn.norm_q",
            "blk.0.attn.norm_k",
            "blk.0.ff.net.2",
        ]
        ranks = s05_schema_diff.rank_breakdown(tensors, "t", modules)
        self.assertEqual(ranks["1-D normalization vectors"], 2)
        self.assertEqual(ranks["2-D linear"], 1)

    def test_checkpoint_resolves_inside_a_directory(self):
        """--quantized-torch-ckpt-save-path is a directory, mto.restore wants a file.

        Handing mto.restore the directory fails with a bare IsADirectoryError
        that explains nothing, so the path is resolved before it gets there.
        """
        from stages import s04_serve_verify

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "torch"
            root.mkdir()
            (root / "metadata.json").write_text("{}")
            big = root / "flux-dev.pt"
            big.write_bytes(b"x" * 4096)

            found = s04_serve_verify.resolve_torch_checkpoint(root)
            self.assertEqual(found.name, "flux-dev.pt")

            # A file path passes straight through.
            self.assertEqual(s04_serve_verify.resolve_torch_checkpoint(big), big)

    def test_checkpoint_resolution_explains_an_empty_directory(self):
        from stages import s04_serve_verify

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "torch"
            root.mkdir()
            (root / "notes.txt").write_text("nothing useful")
            with self.assertRaises(FileNotFoundError) as caught:
                s04_serve_verify.resolve_torch_checkpoint(root)
            self.assertIn("notes.txt", str(caught.exception))

    def test_dynamic_arm_honours_modelopt_exclusions(self):
        """torchao quantizes every Linear unless told otherwise.

        Our first dynamic run had no filter, so it converted the ten layers the
        shipped recipe protects and scored 4.8x further from BF16 than the static
        export. The filter has to protect a root-level name and everything under
        it, while leaving the same name inside a repeated block alone.
        """
        from stages import s02_dynamic_check as s02

        # Protected at the root, and anything beneath it.
        for fqn in (
            "proj_out",
            "x_embedder",
            "context_embedder",
            "time_text_embed.timestep_embedder.linear_1",
            "time_text_embed.guidance_embedder.linear_2",
            "norm_out.linear",
        ):
            self.assertFalse(s02.is_quantizable_path(fqn), f"{fqn} should be excluded")

        # Ordinary GEMMs inside repeated blocks, including ones sharing a leaf name.
        for fqn in (
            "single_transformer_blocks.0.proj_out",
            "transformer_blocks.3.attn.to_q",
            "transformer_blocks.3.norm1.linear",
            "single_transformer_blocks.12.proj_mlp",
        ):
            self.assertTrue(s02.is_quantizable_path(fqn), f"{fqn} should be quantized")

    def test_filter_rejects_an_empty_path(self):
        from stages import s02_dynamic_check as s02

        self.assertFalse(s02.is_quantizable_path(""))

    def test_export_dir_separates_output_from_filter_choice(self):
        """The filter is chosen by model name; the output path must not be.

        To measure what an exclusion filter is worth you quantize the same
        weights under a different filter. If the export directory were derived
        from the model name, that second run would overwrite the first and the
        comparison would be impossible.
        """
        from common import paths as p

        # Normal case: directory follows the model name.
        self.assertEqual(p.export_dir_name({"modelopt_model": "flux-dev"}), "flux-dev")

        # Deliberate mismatch: dev's filter applied to schnell weights.
        self.assertEqual(
            p.export_dir_name(
                {"modelopt_model": "flux-dev", "export_dir": "flux-schnell-devfilter"}
            ),
            "flux-schnell-devfilter",
        )

        # An empty override falls back rather than producing a nameless path.
        self.assertEqual(
            p.export_dir_name({"modelopt_model": "flux-schnell", "export_dir": ""}),
            "flux-schnell",
        )

    def test_quality_recognises_every_nvfp4_arm(self):
        """A new arm name must be added here or its images are silently ignored.

        The quality stage pairs a baseline against any arm in this tuple. When
        `verify` was renamed to emit `nvfp4-static-sim`, an unupdated tuple would
        have meant the static images scored as nothing at all — while the stage
        still printed a result from whatever else it found.
        """
        from stages import s06_quality

        self.assertIn("nvfp4-static-sim", s06_quality.NVFP4_ARMS)
        self.assertIn("nvfp4-dynamic", s06_quality.NVFP4_ARMS)
        self.assertEqual(s06_quality.BF16_ARM, "bf16")
        self.assertNotIn(s06_quality.BF16_ARM, s06_quality.NVFP4_ARMS)

    def test_missing_filter_never_raises(self):
        # MODELOPT_DIFFUSERS_DIR is pointed at a directory that does not exist,
        # so the test covers the branch it names. Reading the real environment
        # meant the assertion exercised whichever path the developer's machine
        # happened to take, and passed either way.
        with mock.patch.dict(
            os.environ, {"MODELOPT_DIFFUSERS_DIR": "/nonexistent-modelopt-checkout"}
        ):
            filter_func, note = s05_schema_diff._load_modelopt_filter()
        self.assertIsNone(filter_func)
        self.assertIsInstance(note, str)
        self.assertTrue(note)

    def test_absent_env_var_is_reported_not_raised(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            filter_func, note = s05_schema_diff._load_modelopt_filter()
        self.assertIsNone(filter_func)
        self.assertIn("MODELOPT_DIFFUSERS_DIR", note)


class TestServedQualityArm(unittest.TestCase):
    """The served arm must be scoreable and must not be confused with the simulated one.

    Both failures here are silent. An arm the quality stage does not recognise
    is skipped without comment, and a directory whose pairing claim is wrong
    invites a reader to over-trust a per-image PSNR.
    """

    def test_served_arm_is_scored(self):
        from stages import s06_quality

        self.assertIn("nvfp4-static-served", s06_quality.NVFP4_ARMS)

    def test_served_and_simulated_are_distinct_arms(self):
        from stages import s06_quality

        self.assertIn("nvfp4-static-sim", s06_quality.NVFP4_ARMS)
        self.assertNotEqual("nvfp4-static-sim", "nvfp4-static-served")

    def test_pairing_defaults_to_injected_latents(self):
        from common import images

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metadata.json"
            spec = images.GenerationSpec(1024, 1024, 50, 3.5, 512, (42,))
            images.write_metadata(path, spec, [])
            self.assertEqual(
                json.loads(path.read_text())["pairing"], images.INJECTED_LATENT_PAIRING
            )

    def test_seeded_pairing_is_recorded_when_asked_for(self):
        """A served directory must say so, because the evidence is weaker."""
        from common import images

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metadata.json"
            spec = images.GenerationSpec(1024, 1024, 50, 3.5, 512, (42,))
            images.write_metadata(path, spec, [], pairing=images.SEEDED_PAIRING)
            pairing = json.loads(path.read_text())["pairing"]
            self.assertNotEqual(pairing, images.INJECTED_LATENT_PAIRING)
            self.assertIn("seed", pairing.lower())


class TestServingCompatFix(unittest.TestCase):
    """The out_channels edit that lets the packed export load in VisualGen.

    Worth testing rather than trusting, because it is applied to a file inside
    an export directory. If a clean run stopped applying it, the symptom would
    be a TypeError deep in a TensorRT-LLM traceback with nothing pointing back
    to here.
    """

    @staticmethod
    def _export(tmp: str, **transformer_config) -> Path:
        hf_dir = Path(tmp) / "hf"
        (hf_dir / "transformer").mkdir(parents=True)
        (hf_dir / "transformer" / "config.json").write_text(json.dumps(transformer_config))
        return hf_dir

    def test_null_out_channels_is_filled_from_in_channels(self):
        with tempfile.TemporaryDirectory() as tmp:
            hf_dir = self._export(tmp, in_channels=64, out_channels=None, patch_size=1)
            result = s03_static_export.materialize_out_channels(hf_dir)

            self.assertTrue(result["changed"])
            self.assertEqual(result["out_channels"], 64)

            written = json.loads((hf_dir / "transformer" / "config.json").read_text())
            self.assertEqual(written["out_channels"], 64)
            # Everything else must survive untouched -- this is a one-field edit.
            self.assertEqual(written["in_channels"], 64)
            self.assertEqual(written["patch_size"], 1)

    def test_original_config_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            hf_dir = self._export(tmp, in_channels=64, out_channels=None)
            s03_static_export.materialize_out_channels(hf_dir)

            backup = hf_dir / "transformer" / "config.json.orig"
            self.assertTrue(backup.is_file(), "the unmodified export must remain recoverable")
            self.assertIsNone(json.loads(backup.read_text())["out_channels"])

    def test_is_idempotent(self):
        """A second export, or a re-run, must not stack edits or lose the backup."""
        with tempfile.TemporaryDirectory() as tmp:
            hf_dir = self._export(tmp, in_channels=64, out_channels=None)
            s03_static_export.materialize_out_channels(hf_dir)
            second = s03_static_export.materialize_out_channels(hf_dir)

            self.assertFalse(second["changed"])
            self.assertEqual(second["reason"], "already set")
            backup = json.loads((hf_dir / "transformer" / "config.json.orig").read_text())
            self.assertIsNone(backup["out_channels"], "backup must stay the pre-edit version")

    def test_explicit_value_is_never_overwritten(self):
        """If a future Model Optimizer emits a real value, leave it alone."""
        with tempfile.TemporaryDirectory() as tmp:
            hf_dir = self._export(tmp, in_channels=64, out_channels=128)
            result = s03_static_export.materialize_out_channels(hf_dir)

            self.assertFalse(result["changed"])
            written = json.loads((hf_dir / "transformer" / "config.json").read_text())
            self.assertEqual(written["out_channels"], 128)

    def test_missing_config_reports_rather_than_raises(self):
        """An export stage that already succeeded must not fail on this."""
        with tempfile.TemporaryDirectory() as tmp:
            result = s03_static_export.materialize_out_channels(Path(tmp) / "absent")
            self.assertFalse(result["changed"])
            self.assertIn("no ", result["reason"])

    def test_both_channels_null_is_reported_not_guessed(self):
        with tempfile.TemporaryDirectory() as tmp:
            hf_dir = self._export(tmp, in_channels=None, out_channels=None)
            result = s03_static_export.materialize_out_channels(hf_dir)
            self.assertFalse(result["changed"])
            self.assertIn("in_channels", result["reason"])


if __name__ == "__main__":
    unittest.main()
