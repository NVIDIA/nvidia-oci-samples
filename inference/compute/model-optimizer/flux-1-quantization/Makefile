# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

.PHONY: help stages test check preflight docx figures bench serve-quality

help:
	@echo "make stages      List the quantization stages and their containers"
	@echo "make preflight   Run environment checks only, no GPU work"
	@echo "make test        Run repository unit tests"
	@echo "make check       Validate a completed run's manifest"
	@echo "make figures     Build BF16-vs-NVFP4 comparison figures from a run"
	@echo "make bench       Time the served NVFP4 export against BF16 (needs the TRT-LLM container)"
	@echo "make serve-quality  Generate and score the paired set from the SERVED checkpoint"

# Comparison figures. Written to docs/figures/ and never committed: this repo is
# public and FLUX.1-dev output is non-commercial. Regenerate locally instead.
#
#   make figures                                    # scores the dynamic arm
#   make figures IMAGES=$$FLUX_QUANT_WORKSPACE/images/verify/flux-schnell
#
# The static arm is the deliverable, so IMAGES pointed at a verify directory is
# usually what you want; that directory carries its own BF16 baseline.
WORKSPACE ?= $(FLUX_QUANT_WORKSPACE)
IMAGES ?= $(WORKSPACE)/images/dynamic

figures:
	@test -n "$(WORKSPACE)" || { echo "set FLUX_QUANT_WORKSPACE or pass WORKSPACE=<dir>"; exit 1; }
	python3 tools/make_figures.py \
		--images "$(IMAGES)" \
		--results "$(WORKSPACE)/results" \
		--sheet-seed 31415

# Optional, and only present in a working copy that has a local operator runbook.
# RUNBOOK.md is site-specific and gitignored, so this is a no-op after a clean
# clone rather than an error.
docx:
	@# One recipe line on purpose. `exit 0` in a guard only ends that line, so a
	@# multi-line version still ran pandoc and failed -- and RUNBOOK.md is
	@# gitignored, so a clean clone is exactly the path the guard promises to skip.
	@if [ ! -f RUNBOOK.md ]; then \
	  echo "No RUNBOOK.md in this working copy; nothing to build."; \
	elif ! command -v pandoc >/dev/null 2>&1; then \
	  echo "pandoc not found: brew install pandoc"; exit 1; \
	else \
	  pandoc RUNBOOK.md -f markdown -t docx -o RUNBOOK.docx && echo "Wrote RUNBOOK.docx."; \
	fi

# Serving benchmark. Must run inside the TensorRT-LLM container, which is where
# VisualGen lives; the script checks and says so rather than failing obscurely.
MODEL ?= all

bench:
	./scripts/bench-serving.sh $(MODEL)

# Quality from the packed export on real kernels, rather than from mto.restore
# simulation. Both arms go through VisualGen so precision is the only variable.
serve-quality:
	./scripts/serve-quality.sh $(MODEL)

stages:
	python3 quantize.py --list-stages

preflight:
	python3 quantize.py --stage preflight

test:
	python3 -m compileall -q quantize.py common stages tools tests
	python3 -m unittest discover -s tests -v

check:
	python3 tools/check_run_manifest.py
