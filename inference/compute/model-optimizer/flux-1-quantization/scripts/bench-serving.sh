#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Benchmark the packed NVFP4 export against BF16 through TensorRT-LLM VisualGen.
#
#   ./scripts/bench-serving.sh flux-dev
#   ./scripts/bench-serving.sh flux-schnell
#   ./scripts/bench-serving.sh all
#
# This answers the one question the quality work never could. `mto.restore`
# reproduces NVFP4 *numerics* on BF16 storage, so it cannot say anything about
# speed; only the packed `--hf-ckpt-dir` export on real kernels can.
#
# Results go to results/serving-bench-<model>.json, one file per model on
# purpose. flux-dev runs 50 steps and flux-schnell runs 4, so a single table
# would show a 12x "speedup" that is entirely step count.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

TARGET="${1:-all}"

# ---------------------------------------------------------------- preconditions

if ! python3 -c "import tensorrt_llm" 2>/dev/null; then
  cat >&2 <<'MSG'
STOP: TensorRT-LLM is not importable, so nothing here can run.

VisualGen lives in the TRT-LLM container. Attach to it first:

  enroot start --rw \
    --mount /path/to/scratch:/path/to/scratch trtllm bash
MSG
  exit 1
fi

WORKSPACE="${FLUX_QUANT_WORKSPACE:-}"
if [[ -z "$WORKSPACE" ]]; then
  echo "STOP: set FLUX_QUANT_WORKSPACE to the run workspace." >&2
  exit 1
fi

# Ask the harness where checkpoints live rather than guessing from $WORKSPACE.
# The shared models directory sits on the scratch volume, which is not
# necessarily the workspace's parent -- deriving it with `dirname` silently
# produces a path that does not exist, and every stock-model arm then skips
# while the static arms still run, which looks like a partial result rather
# than a broken script.
MODELS_DIR="$(python3 -c '
import sys
sys.path.insert(0, ".")
from common import paths
print(paths.resolve(create=False).models)
' 2>/dev/null || true)"

if [[ -z "$MODELS_DIR" || ! -d "$MODELS_DIR" ]]; then
  echo "STOP: could not resolve the models directory (got '${MODELS_DIR:-<empty>}')." >&2
  echo "      Set FLUX_QUANT_MODELS to the directory holding FLUX.1-dev and FLUX.1-schnell." >&2
  exit 1
fi
echo "  models:    $MODELS_DIR"
echo "  workspace: $WORKSPACE"

RESULTS="$WORKSPACE/results"
IMAGES="$WORKSPACE/images/served"
CONFIGS="$WORKSPACE/configs"
mkdir -p "$RESULTS" "$IMAGES" "$CONFIGS"

# ------------------------------------------------------------- serving config
#
# No quant_config in this YAML, deliberately. VisualGen has two ways to learn
# that a model is quantized and they disagree by design:
#
#   * config_groups in the checkpoint  -> `dynamic` defaults to FALSE (static)
#   * quant_config in this YAML        -> `dynamic` defaults to TRUE  (runtime)
#
# Our export declares "dynamic": false. Supplying quant_config here would
# override it and re-quantize at load time, and the only symptom would be a
# speedup that never arrives. Leaving it out lets the checkpoint speak.

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

DYNAMIC_YAML=/app/tensorrt_llm/examples/visual_gen/configs/flux1-dev-fp4-1gpu.yaml

# ---------------------------------------------------------------------- helpers

MISSING=()

require() {  # require <label> <path> <kind>
  local label="$1" path="$2" kind="$3"

  if [[ ! -e "$path" ]]; then
    printf '  %-34s %-8s MISSING  %s\n' "$label" "$kind" "$path"
    MISSING+=("$label -> $path")
    return
  fi

  case "$kind" in
    pipeline)
      # A download interrupted partway leaves the directory structure and the
      # metadata but no weights, so existence alone is not enough.
      if [[ ! -f "$path/model_index.json" ]]; then
        printf '  %-34s %-8s BROKEN   no model_index.json in %s\n' "$label" "$kind" "$path"
        MISSING+=("$label -> $path (no model_index.json)")
        return
      fi
      ;;
    export)
      # The packed export must carry an explicit out_channels or the load dies
      # inside transformer_flux.py with a TypeError that names nothing useful.
      # The export stage writes it; this catches an export made before that
      # landed, which is the exact failure this script existed to hit once.
      # The path goes through argv, not into the Python source. Interpolated,
      # a workspace path containing a quote or a backslash became a SyntaxError
      # that this function swallowed and reported as "BROKEN out_channels=ERR"
      # -- pointing at the checkpoint instead of at the path.
      local oc
      oc="$(python3 -c '
import json, sys
try:
    with open(sys.argv[1] + "/transformer/config.json") as handle:
        print(json.load(handle).get("out_channels"))
except Exception as exc:
    print("ERR", exc)
' "$path" 2>/dev/null || echo ERR)"
      if [[ "$oc" == "None" || "$oc" == ERR* || -z "$oc" ]]; then
        printf '  %-34s %-8s BROKEN   out_channels=%s, run the export stage or apply the fix\n' \
               "$label" "$kind" "$oc"
        MISSING+=("$label -> $path (out_channels=$oc)")
        return
      fi
      ;;
  esac

  printf '  %-34s %-8s ok\n' "$label" "$kind"
}

preflight() {
  echo "=============================================================="
  echo " Preflight - checking every input before any GPU work"
  echo "=============================================================="

  require "shipped dynamic config" "$DYNAMIC_YAML" file

  if [[ "$TARGET" == "flux-dev" || "$TARGET" == "all" ]]; then
    require "FLUX.1-dev (BF16 baseline)"  "$MODELS_DIR/FLUX.1-dev"          pipeline
    require "flux-dev packed export"      "$WORKSPACE/exports/flux-dev/hf"  export
  fi

  if [[ "$TARGET" == "flux-schnell" || "$TARGET" == "all" ]]; then
    require "FLUX.1-schnell (BF16 baseline)"  "$MODELS_DIR/FLUX.1-schnell"              pipeline
    require "flux-schnell packed export"      "$WORKSPACE/exports/flux-schnell/hf"      export
    require "flux-schnell-devfilter export"   "$WORKSPACE/exports/flux-schnell-devfilter/hf" export
  fi

  if (( ${#MISSING[@]} )); then
    echo
    echo "STOP: ${#MISSING[@]} input(s) are missing or unusable:"
    printf '   %s\n' "${MISSING[@]}"
    cat <<'MSG'

Nothing has been run. Fix these first -- a partial benchmark is worse than
none, because the arms that do work still write numbers and a table missing
its BF16 row reads like a result.

  missing baseline  -> python3 quantize.py --stage download
  missing export    -> python3 quantize.py --stage export --force
  out_channels=None -> re-run the export stage, which now writes it
MSG
    exit 1
  fi
  echo "  all inputs present"
}

SKIPPED=()

bench() {  # bench <out-json> <name> <model> <steps> <guidance> [extra args...]
  local out="$1" name="$2" model="$3" steps="$4" guidance="$5"; shift 5
  if [[ ! -e "$model" ]]; then
    # A skipped baseline is worse than a crash: the remaining arms still record
    # numbers, and a table with no BF16 row looks like a result rather than a
    # broken run. Collect these and fail loudly at the end.
    echo "  SKIP $name - nothing at $model" >&2
    SKIPPED+=("$name -> $model")
    return 0
  fi
  echo
  echo "--- $name ---"
  python3 tools/bench_serving.py \
    --name "$name" --model "$model" \
    --steps "$steps" --guidance-scale "$guidance" \
    --out "$out" --save-image "$IMAGES/bench-$name.png" "$@"
}

run_flux_dev() {
  local out="$RESULTS/serving-bench-flux-dev.json"
  # bench_serving.py only ever appends. Without this a re-run stacks the new
  # arms on top of the previous run's, and the summary's baseline lookup takes
  # the *first* matching bf16 record -- one measured against a possibly
  # different driver, checkpoint or export.
  rm -f "$out"
  echo "=============================================================="
  echo " flux-dev, 50 steps - precision held against a constant model"
  echo "=============================================================="
  # Same weights in every arm; only the precision path changes, so the
  # difference is attributable to quantization rather than to the model.
  bench "$out" "dev-bf16"          "$MODELS_DIR/FLUX.1-dev"        50 3.5
  bench "$out" "dev-nvfp4-dynamic" "$MODELS_DIR/FLUX.1-dev"        50 3.5 \
        --visual-gen-args "$DYNAMIC_YAML"
  bench "$out" "dev-nvfp4-static"  "$WORKSPACE/exports/flux-dev/hf" 50 3.5 \
        --visual-gen-args "$STATIC_YAML"
  echo; echo "  -> $out"
}

run_flux_schnell() {
  local out="$RESULTS/serving-bench-flux-schnell.json"
  # Same reason as flux-dev. The devfilter load-check writes its own file and
  # accumulates identically, so it is reset here too.
  rm -f "$out" "$RESULTS/serving-loadcheck.json"
  echo
  echo "=============================================================="
  echo " flux-schnell, 4 steps - the Apache-2.0 arm"
  echo "=============================================================="
  # schnell is the only arm whose images may be published or shared, so a
  # serving result here matters more than dev's for anything customer-facing.
  bench "$out" "schnell-bf16"          "$MODELS_DIR/FLUX.1-schnell"        4 0.0
  bench "$out" "schnell-nvfp4-dynamic" "$MODELS_DIR/FLUX.1-schnell"        4 0.0 \
        --visual-gen-args "$DYNAMIC_YAML"
  bench "$out" "schnell-nvfp4-static"  "$WORKSPACE/exports/flux-schnell/hf" 4 0.0 \
        --visual-gen-args "$STATIC_YAML"

  # Load check, not a benchmark. devfilter differs from plain schnell by one
  # quantized layer (494 vs 495), so timing them apart would measure noise --
  # but it is needed for the served quality comparison, and finding out then
  # that it does not open would cost an allocation.
  echo
  echo "--- devfilter load check (1 iteration, no warm-up) ---"
  bench "$RESULTS/serving-loadcheck.json" "schnell-devfilter-loadcheck" \
        "$WORKSPACE/exports/flux-schnell-devfilter/hf" 4 0.0 \
        --visual-gen-args "$STATIC_YAML" --warmup 0 --iterations 1
  echo; echo "  -> $out"
}

# ------------------------------------------------------------------------- main

case "$TARGET" in
  flux-dev|flux-schnell|all) ;;
  *)
    echo "usage: $0 [flux-dev|flux-schnell|all]" >&2
    exit 2
    ;;
esac

preflight
echo

case "$TARGET" in
  flux-dev)     run_flux_dev ;;
  flux-schnell) run_flux_schnell ;;
  all)          run_flux_dev; run_flux_schnell ;;
esac

echo
echo "=============================================================="
echo " Summary"
echo "=============================================================="
python3 - "$RESULTS" <<'PY'
import json, sys
from pathlib import Path

results = Path(sys.argv[1])
for path in sorted(results.glob("serving-bench-*.json")):
    try:
        arms = json.loads(path.read_text())
    except Exception as error:
        print(f"{path.name}: unreadable ({error})")
        continue

    print(f"\n{path.name}")
    print(f"  {'arm':<26} {'median s':>9} {'ms/step':>9} {'vs BF16':>9}")
    baseline = next((a["median_s"] for a in arms if a["arm"].endswith("bf16")), None)
    for arm in arms:
        speedup = f"{baseline / arm['median_s']:.2f}x" if baseline else "-"
        print(f"  {arm['arm']:<26} {arm['median_s']:>9.2f} "
              f"{arm['s_per_step'] * 1000:>9.0f} {speedup:>9}")

    # A static arm that lands on BF16 latency is the failure this whole script
    # exists to catch: the checkpoint loaded, generated fine, and never used a
    # 4-bit kernel. Say so rather than leaving it to be read off the table.
    for arm in arms:
        if "static" in arm["arm"] and baseline and arm["median_s"] > baseline * 0.95:
            print(f"  WARNING: {arm['arm']} is within 5% of BF16. Check that the "
                  f"checkpoint was read as quantized rather than re-quantized.")

    if baseline is None:
        print("  WARNING: no BF16 arm in this file. Speedups cannot be computed, "
              "and the absolute latencies mean nothing on their own.")
PY

if (( ${#SKIPPED[@]} )); then
  echo
  echo "!! ${#SKIPPED[@]} arm(s) were skipped because the model was not found:"
  printf '     %s\n' "${SKIPPED[@]}"
  echo "   The numbers above are incomplete. Fix the paths and re-run before"
  echo "   quoting anything from them."
  exit 1
fi
