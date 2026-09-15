#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --time=00:20:00
#SBATCH --job-name=flux-bench-smoke
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

: "${MODEL_DIR:?Set MODEL_DIR to a local FLUX.1-schnell Diffusers checkpoint}"

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:?Submit this job from the repository root or set REPO_ROOT}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/results/smoke/$SLURM_JOB_ID}"
CACHE_ROOT="${CACHE_ROOT:-${TMPDIR:-/tmp}/flux-bench-$SLURM_JOB_ID}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

mkdir -p "$OUTPUT_ROOT/results" "$CACHE_ROOT/home" "$CACHE_ROOT/xdg" "$CACHE_ROOT/huggingface"

srun_args=(
  --nodes=1
  --ntasks=1
  --cpus-per-task="${SLURM_CPUS_PER_TASK:-8}"
)

if [[ -n "${CONTAINER_IMAGE:-}" ]]; then
  default_mounts="$REPO_ROOT:$REPO_ROOT,$MODEL_DIR:$MODEL_DIR,$CACHE_ROOT:$CACHE_ROOT"
  srun_args+=(
    --container-image="$CONTAINER_IMAGE"
    --container-mounts="${CONTAINER_MOUNTS:-$default_mounts}"
    --container-workdir="$REPO_ROOT"
    --no-container-mount-home
    --no-container-entrypoint
  )
fi

run_step() {
  srun "${srun_args[@]}" \
    env \
      HOME="$CACHE_ROOT/home" \
      XDG_CACHE_HOME="$CACHE_ROOT/xdg" \
      HF_HOME="${HF_HOME:-$CACHE_ROOT/huggingface}" \
      HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" \
      TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}" \
      PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
      "$@"
}

run_step "$PYTHON_BIN" "$REPO_ROOT/benchmark.py" \
  --mode hf-diffusers-bf16-compile \
  --model "$MODEL_DIR" \
  --batches 1 \
  --warmup 1 \
  --iterations 2 \
  --output-dir "$OUTPUT_ROOT/results"

result="$OUTPUT_ROOT/results/hf-diffusers/hf-diffusers-bf16-compile/b1.json"
run_step "$PYTHON_BIN" -m tools.check_result_contract \
  --expected-batch 1 \
  --expected-warmup 1 \
  --expected-iterations 2 \
  "$result"

echo "RESULT=$result"
