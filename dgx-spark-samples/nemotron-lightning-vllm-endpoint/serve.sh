#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Serve Nemotron 3.5 Lightning 30B-A3B (NVFP4) on DGX Spark via vLLM.
#
# First start is ~4 minutes; later starts are quicker once the compile and
# autotune caches are warm. Leave this running in its own terminal; the
# notebook talks to it over HTTP.
#
# Usage:
#   ./serve.sh              # foreground
#   ./serve.sh --detach     # background container, logs via: docker logs -f vllm-nemotron
#   SKIP_PRECHECK=1 ./...   # skip the architecture-support probe
#   TUNED=1 ./serve.sh      # add NVIDIA's DGX Spark tuning flags (not the
#                           # configuration the published numbers came from)

set -euo pipefail

MODEL="${MODEL:-nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4}"
PORT="${PORT:-8000}"
# Which host interface the published port is bound to. Loopback by default:
# vLLM serves no authentication of any kind, so publishing on every interface
# puts an open completions endpoint -- and the GPU behind it -- on whatever
# network the Spark is attached to. The notebook runs on the device itself, so
# the default costs it nothing.
#
# To reach the endpoint from a laptop, prefer the SSH tunnel documented in
# README.md:
#
#     ssh -N -L 8000:localhost:8000 <user>@<device>.local
#
# If you genuinely need it on the network -- a trusted lab segment, say -- opt
# in explicitly and understand that anyone who can route to this host can use
# the model:
#
#     BIND_ADDR=0.0.0.0 ./serve.sh
BIND_ADDR="${BIND_ADDR:-127.0.0.1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
CONTAINER_NAME="${CONTAINER_NAME:-vllm-nemotron}"
# v0.27.1 is the release that carries day-0 Nemotron 3.5 Lightning support, and
# it is the tag NVIDIA and the vLLM team publish for this model. Older releases
# fail at config load with "model type `nemotron_h` but Transformers does not
# recognize this architecture" -- see Known Issues in README.md.
#
# vLLM also publishes arch-suffixed tags (`*-aarch64`), but this one is a
# multi-arch manifest list containing linux/arm64, so Docker resolves the right
# platform on GB10 automatically. Pinned rather than a nightly, which can be
# pruned from the registry without warning.
#
# Known-good history on this hardware: the published run used the nightly
# vLLM 0.26.1rc1.dev403. v0.27.1 is the later stable release of that lineage and
# is what a customer can still pull today, which a nightly tag is not. If
# v0.27.1 ever misbehaves and you still have the old nightly in your local
# Docker cache (check: docker images | grep vllm), you can fall back with
# VLLM_IMAGE=<that tag> ./serve.sh
IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:v0.27.1}"
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"
# vLLM writes its torch.compile artefacts and FlashInfer autotune results to
# /root/.cache/vllm. The container is --rm, so without a mount that work is
# thrown away and redone on every start: ~7s of compilation plus ~18s of fp8
# GEMM autotuning, every time. Persist it and only the first start pays.
VLLM_CACHE="${VLLM_CACHE_DIR:-$HOME/.cache/vllm}"

DETACH_FLAG=""
if [[ "${1:-}" == "--detach" ]]; then
  DETACH_FLAG="-d"
fi

# --- preflight ---------------------------------------------------------------

if ! command -v docker >/dev/null 2>&1; then
  echo "ERROR: docker not found." >&2
  exit 1
fi

if ! docker info >/dev/null 2>&1; then
  cat >&2 <<'EOF'
ERROR: cannot talk to the Docker daemon.

  sudo gpasswd -a $(whoami) docker

Then log out and back in. If you are in a JupyterLab terminal, restart the
JupyterLab *server* -- a kernel restart does not pick up new group membership.
Use $(whoami), not $USER: in a Jupyter-spawned shell $USER can be 'root'.
EOF
  exit 1
fi

# IMPORTANT: mount the whole HF cache, not just the snapshot directory.
# Snapshot dirs are symlink farms into blobs/ -- mounting only the snapshot
# leaves every link dangling and the model fails to load.
mkdir -p "$HF_CACHE" "$VLLM_CACHE"

if docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
  echo "Removing existing container '$CONTAINER_NAME'..."
  docker rm -f "$CONTAINER_NAME" >/dev/null
fi

# Is the port already taken by something we did not start? Docker's own error
# for this ("address already in use") does not say what is holding it, which
# leaves you guessing. Check first and say something useful.
port_holder() {
  docker ps --format '{{.Names}}\t{{.Ports}}' 2>/dev/null \
    | awk -v p=":${PORT}->" '$0 ~ p {print "docker container: " $1}'
}

if command -v ss >/dev/null 2>&1 && ss -tln 2>/dev/null | grep -q ":${PORT} "; then
  HOLDER="$(port_holder)"
  cat >&2 <<EOF

ERROR: port ${PORT} is already in use.

$( [[ -n "$HOLDER" ]] && echo "  Held by ${HOLDER}" || echo "  Not a container this script manages. Identify it with:
    docker ps --format '{{.Names}}\t{{.Ports}}'
    sudo ss -tlnp | grep :${PORT}" )

If it is already serving this model, you do not need to start another one:

    curl -s http://localhost:${PORT}/v1/models | python3 -m json.tool

Otherwise either stop it, or run on a different port:

    PORT=8001 ./serve.sh

If you change the port, update BASE_URL in demo.ipynb to match.
EOF
  exit 1
fi

# Can this image actually load this model? vLLM compiles architecture support
# in; it is not carried by the checkpoint. An image that predates the model
# fails ~10s into startup with a pydantic ValidationError about Transformers not
# recognising the architecture, which sends people off upgrading Transformers --
# the wrong fix, and a slow way to find that out.
#
# So reproduce that exact config load, offline against the cached weights,
# before committing to a five-minute wait.
#
# Two rules this probe follows, learned by getting it wrong:
#
#   1. Do not infer support from vLLM's own _CONFIG_REGISTRY. That registry is
#      the *fallback* for architectures Transformers does not know. A model
#      Transformers handles natively is absent from it and works fine, so an
#      absence test reports healthy images as broken.
#   2. Report "no" only on the specific architecture-recognition error. Any
#      other failure -- API drift, no cached weights, no network -- is
#      inconclusive, and an inconclusive probe must not block a working setup.
if [[ -z "${SKIP_PRECHECK:-}" && -d "$HF_CACHE" ]]; then
  PROBE="$(docker run --rm \
    -v "${HF_CACHE}:/root/.cache/huggingface" \
    -e HF_HUB_OFFLINE=1 \
    --entrypoint python3 "$IMAGE" -c "
import sys
SENTINEL = 'does not recognize this architecture'
msg = ''
try:
    from vllm.transformers_utils.config import get_config
    get_config('${MODEL}', trust_remote_code=False)
    print('yes'); sys.exit()
except TypeError:
    pass                      # signature drift across releases; fall through
except Exception as e:
    msg = str(e)
if SENTINEL in msg:
    print('no'); sys.exit()
try:
    from transformers import AutoConfig
    AutoConfig.from_pretrained('${MODEL}')
    print('yes')
except Exception as e:
    print('no' if SENTINEL in str(e) else 'unknown')
" 2>/dev/null | tail -1)"

  if [[ "$PROBE" == "no" ]]; then
    cat >&2 <<EOF

ERROR: this vLLM image cannot load this model's architecture.

  Image: ${IMAGE}
  Model: ${MODEL}

It would start, read the config, and fail about ten seconds in with:

  Value error, The checkpoint you are trying to load has model type
  \`nemotron_h\` but Transformers does not recognize this architecture.

Ignore that message's advice. Upgrading Transformers will not fix it, and
--trust-remote-code cannot substitute: this checkpoint has no auto_map, so
there is no remote code to load. Architecture support is compiled into vLLM.

Use a newer vLLM release -- the tag NVIDIA publishes for this model is
v0.27.1, and anything later should also carry it:

    VLLM_IMAGE=vllm/vllm-openai:<newer-tag> ./serve.sh

To bypass this check entirely:

    SKIP_PRECHECK=1 ./serve.sh

EOF
    exit 1
  elif [[ "$PROBE" != "yes" ]]; then
    echo "NOTE: could not verify architecture support in this image (probe said"
    echo "      '${PROBE:-nothing}'). Continuing -- this is not a failure signal."
    echo
  fi
fi

echo "Model:        $MODEL"
echo "Port:         $PORT"
if [[ "$BIND_ADDR" == "127.0.0.1" || "$BIND_ADDR" == "localhost" ]]; then
  echo "Bind:         $BIND_ADDR (device-local only; use an SSH tunnel from a laptop)"
else
  echo "Bind:         $BIND_ADDR  ** reachable from the network, and vLLM has no auth **"
fi
echo "Context:      $MAX_MODEL_LEN"
echo "HF cache:     $HF_CACHE"
echo "vLLM cache:   $VLLM_CACHE"
echo "Image:        $IMAGE"
echo
echo "First start takes ~4 minutes; later starts are quicker once the compile\nand autotune caches are warm. Endpoint will be at http://localhost:$PORT/v1"
echo

# --- serve -------------------------------------------------------------------
#
# THE DEFAULT HERE IS THE CONFIGURATION THAT HAS ACTUALLY BEEN RUN.
#
# Four flags, and no more. This is what produced the published When2Call results
# on this hardware: 120/120 examples scored, zero errors, 17.86 GiB of weights
# resident and 84.78 GiB left for KV cache. Every flag below earns its place.
#
#   --max-model-len 65536
#       64k context. The model supports up to 1M; 64k is what was measured.
#   --reasoning-parser nemotron_v3
#       returns the reasoning trace as a structured field instead of inline text
#   --enable-auto-tool-choice --tool-call-parser qwen3_coder
#       returns tool calls as structured tool_calls instead of text to scrape
#
# Deliberately NOT passed:
#
#   --kv-cache-dtype fp8   This ModelOpt checkpoint pins fp8 KV in its own
#                          quantisation config, so vLLM selects fp8_e4m3 whether
#                          or not you pass it. Verified on this machine. Passing
#                          it implies a choice that is not yours to make.
#   --moe-backend marlin   GB10 has no native FP4 compute, so vLLM already falls
#                          back to Marlin on its own and logs that it did.
#                          Pinning it by hand adds nothing and would silently
#                          override a better default on other hardware.
#   --trust-remote-code    The checkpoint has no auto_map. There is no remote
#                          code to trust, so this only widens what would run if
#                          the repo were ever compromised.
#
# TUNED=1 opts into NVIDIA's fuller DGX Spark configuration from the vLLM day-0
# blog: mamba cache tuning, prefix caching, explicit backends. It is a
# reasonable configuration, published by people who know this model. It is also
# not the one these numbers came from, and two of its flags change what you are
# measuring -- see README, "Tuned profile". Benchmark on the default; explore
# with TUNED=1.

SERVE_FLAGS=(
  --max-model-len "$MAX_MODEL_LEN"
  --reasoning-parser nemotron_v3
  --enable-auto-tool-choice
  --tool-call-parser qwen3_coder
)

if [[ -n "${TUNED:-}" ]]; then
  echo "TUNED=1: adding NVIDIA's DGX Spark tuning flags."
  echo "         Results are not comparable to a default-profile run."
  echo
  SERVE_FLAGS+=(
    # --trust-remote-code is deliberately NOT here. The supported checkpoint
    # ships no `auto_map`, so it needs no custom code path, and the flag would
    # let a different or re-tagged model execute repository Python inside the
    # container -- which holds HF_TOKEN and a writable cache mount. If you
    # point MODEL at a checkpoint that genuinely requires it, add it yourself
    # and pin the revision you reviewed.
    --moe-backend marlin
    --kv-cache-dtype fp8
    --mamba-backend flashinfer
    --mamba-ssm-cache-dtype float16
    --enable-mamba-cache-stochastic-rounding
    --mamba-cache-philox-rounds 5
    --mamba-cache-mode align
    --enable-prefix-caching
    --max-num-batched-tokens 16384
  )
fi

# --entrypoint vllm is deliberate, and the failure it prevents is worth knowing.
#
# This image ships ENTRYPOINT ["vllm", "serve"]. Passing "vllm serve <model>" as
# the container command therefore runs:
#
#     vllm serve vllm serve <model> --flags
#            ^^^^ entrypoint    ^^^^ our command
#
# argparse consumes the literal string "vllm" as the model_tag positional and
# then reports the real model as junk:
#
#     vllm: error: unrecognized arguments: serve nvidia/NVIDIA-Nemotron-...
#
# which reads like a bad model name and is not. Overriding the entrypoint makes
# the final command exactly "vllm serve <model> --flags" regardless of what the
# image's default entrypoint is in this or a future tag. Confirm yours with:
#
#     docker inspect "$IMAGE" --format '{{json .Config.Entrypoint}}'

# shellcheck disable=SC2086
exec docker run $DETACH_FLAG --rm \
  --name "$CONTAINER_NAME" \
  --gpus all \
  --ipc=host \
  -p "${BIND_ADDR}:${PORT}:8000" \
  -v "${HF_CACHE}:/root/.cache/huggingface" \
  -v "${VLLM_CACHE}:/root/.cache/vllm" \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  --entrypoint vllm \
  "$IMAGE" \
  serve "$MODEL" \
    "${SERVE_FLAGS[@]}" \
    --host 0.0.0.0 \
    --port 8000
# --host 0.0.0.0 here is the bind *inside* the container, and it has to stay:
# Docker reaches the process over the container's own network interface, so
# binding it to 127.0.0.1 would make the published port unreachable and look
# like a dead server. What limits exposure is -p "${BIND_ADDR}:...", which is
# the host side of the mapping.
