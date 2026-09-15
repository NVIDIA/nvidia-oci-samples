#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Score the packed NVFP4 export as deployed, not as simulated.
#
#   ./scripts/serve-quality.sh flux-dev
#   ./scripts/serve-quality.sh flux-schnell
#   ./scripts/serve-quality.sh flux-schnell-devfilter
#   ./scripts/serve-quality.sh all
#
# Every quality figure before this came from `mto.restore` -- NVFP4 numerics
# executed in BF16. This generates the same prompt set from the packed 4-bit
# export running on real kernels through TensorRT-LLM VisualGen, then scores it.
#
# Both arms go through VisualGen. Reusing the verify stage's BF16 images would
# change pipeline and precision at once, and no difference would be attributable.
#
# Roughly 6 minutes for flux-dev at 50 steps, 2 for flux-schnell at 4.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

TARGET="${1:-all}"
case "$TARGET" in
  flux-dev|flux-schnell|flux-schnell-devfilter|all) ;;
  *) echo "usage: $0 [flux-dev|flux-schnell|flux-schnell-devfilter|all]" >&2; exit 2 ;;
esac

if ! python3 -c "import tensorrt_llm" 2>/dev/null; then
  echo "STOP: TensorRT-LLM is not importable. Attach to the TRT-LLM container first." >&2
  exit 1
fi

WORKSPACE="${FLUX_QUANT_WORKSPACE:-}"
[[ -n "$WORKSPACE" ]] || { echo "STOP: set FLUX_QUANT_WORKSPACE." >&2; exit 1; }

MODELS_DIR="$(python3 -c '
import sys; sys.path.insert(0, ".")
from common import paths
print(paths.resolve(create=False).models)
' 2>/dev/null || true)"
[[ -d "$MODELS_DIR" ]] || { echo "STOP: could not resolve the models directory." >&2; exit 1; }

RESULTS="$WORKSPACE/results"
CONFIGS="$WORKSPACE/configs"
mkdir -p "$RESULTS" "$CONFIGS"

# No quant_config, deliberately. With it, VisualGen takes its YAML path where
# `dynamic` defaults to true and re-quantizes at load time -- which would score
# the runtime-quantized model while claiming to score our checkpoint.
STATIC_YAML="$CONFIGS/flux1-static-1gpu.yaml"
cat > "$STATIC_YAML" <<'YAML'
attention_config:
  backend: VANILLA
parallel_config:
  cfg_size: 1
  ulysses_size: 1
cuda_graph_config:
  enable: false
YAML

MISSING=()
require() {
  local label="$1" path="$2"
  if [[ ! -e "$path" ]]; then
    printf '  %-32s MISSING  %s\n' "$label" "$path"; MISSING+=("$label"); return
  fi
  printf '  %-32s ok\n' "$label"
}

run_model() {  # run_model <name> <config> <baseline-dir> <export-dir>
  local name="$1" config="$2" baseline="$3" export_dir="$4"
  local out="$WORKSPACE/images/served-quality/$name"

  echo
  echo "=============================================================="
  echo " $name - generating both arms through VisualGen"
  echo "=============================================================="

  # One arm per process. VisualGen spawns a worker that holds the model
  # resident and `del` does not reap it, so a second pipeline in the same
  # process would leave the first occupying memory. Two loads, no ambiguity.
  echo
  echo "--- [1/2] bf16 through VisualGen ---"
  python3 tools/serve_generate.py \
    --arm bf16 --model "$baseline" \
    --config "$config" --out "$out" \
    --determinism-check

  echo
  echo "--- [2/2] nvfp4-static-served through VisualGen ---"
  python3 tools/serve_generate.py \
    --arm nvfp4-static-served --model "$export_dir" \
    --config "$config" --out "$out" \
    --visual-gen-args "$STATIC_YAML" \
    --quant-recipe "modelopt static PTQ, packed NVFP4 on TRT-LLM VisualGen" \
    --determinism-check

  echo
  echo "--- scoring $name ---"
  python3 quantize.py --stage quality --force --images "$out"
}

echo "=============================================================="
echo " Preflight"
echo "=============================================================="
if [[ "$TARGET" == "flux-dev" || "$TARGET" == "all" ]]; then
  require "FLUX.1-dev baseline"      "$MODELS_DIR/FLUX.1-dev"
  require "flux-dev packed export"   "$WORKSPACE/exports/flux-dev/hf"
fi
if [[ "$TARGET" == "flux-schnell" || "$TARGET" == "all" ]]; then
  require "FLUX.1-schnell baseline"    "$MODELS_DIR/FLUX.1-schnell"
  require "flux-schnell packed export" "$WORKSPACE/exports/flux-schnell/hf"
fi
if [[ "$TARGET" == "flux-schnell-devfilter" || "$TARGET" == "all" ]]; then
  require "FLUX.1-schnell baseline"    "$MODELS_DIR/FLUX.1-schnell"
  require "devfilter packed export"    "$WORKSPACE/exports/flux-schnell-devfilter/hf"
fi
# CMMD is the headline metric. Without the checkout, a run still completes and
# still writes PSNR and CLIP, so the absence is easy to miss until someone asks
# for the number the whole comparison turns on. Check it up front instead.
CMMD_FOUND="$(python3 -c '
import sys; sys.path.insert(0, ".")
from pathlib import Path
from common import paths
from stages import s06_quality
print(s06_quality._find_cmmd_repo(paths.resolve(create=False).root) or "")
' 2>/dev/null || true)"
if [[ -n "$CMMD_FOUND" ]]; then
  printf '  %-32s ok  %s\n' "cmmd-pytorch" "$CMMD_FOUND"
else
  printf '  %-32s MISSING\n' "cmmd-pytorch"
  MISSING+=("cmmd-pytorch — set CMMD_REPO to a checkout of github.com/sayakpaul/cmmd-pytorch")
fi

if (( ${#MISSING[@]} )); then
  echo; echo "STOP: ${#MISSING[@]} input(s) missing. Nothing has been run." >&2
  printf '   %s\n' "${MISSING[@]}" >&2
  exit 1
fi
echo "  all inputs present"

[[ "$TARGET" == "flux-dev"     || "$TARGET" == "all" ]] && run_model flux-dev \
  configs/flux-dev.json "$MODELS_DIR/FLUX.1-dev" "$WORKSPACE/exports/flux-dev/hf"
[[ "$TARGET" == "flux-schnell" || "$TARGET" == "all" ]] && run_model flux-schnell \
  configs/flux-schnell.json "$MODELS_DIR/FLUX.1-schnell" "$WORKSPACE/exports/flux-schnell/hf"

# schnell's weights under flux-dev's exclusion filter. This is the variant whose
# layers verify (filter_agrees: true) and the one the published CMMD 0.026 refers
# to, so it is the arm the served figure must be compared against.
[[ "$TARGET" == "flux-schnell-devfilter" || "$TARGET" == "all" ]] && run_model flux-schnell-devfilter \
  configs/flux-schnell-devfilter.json "$MODELS_DIR/FLUX.1-schnell" \
  "$WORKSPACE/exports/flux-schnell-devfilter/hf"

echo
echo "=============================================================="
echo " Done"
echo "=============================================================="
echo "  images:  $WORKSPACE/images/served-quality/<model>/"
echo "  scores:  $RESULTS/quality-<model>.json"
echo
echo "  Compare against the mto.restore figures already on the page."
echo "  A large gap between simulated and served would itself be the finding."
