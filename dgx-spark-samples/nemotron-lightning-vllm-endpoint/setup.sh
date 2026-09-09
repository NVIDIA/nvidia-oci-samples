#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# One-time setup: verify the machine, install client deps, pre-download weights.
#
# Run this once. Then ./serve.sh to start the endpoint.
# Downloads ~20 GB of model weights on a first run.

set -euo pipefail

MODEL="${MODEL:-nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4}"
IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:v0.27.1}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

say() { printf '\n\033[1m== %s\033[0m\n' "$1"; }

# --- 1. machine check --------------------------------------------------------

say "Checking the machine"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "WARNING: nvidia-smi not found. Is this a DGX Spark with the NVIDIA stack installed?"
else
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
fi

ARCH="$(uname -m)"
echo "Architecture: $ARCH"
if [[ "$ARCH" != "aarch64" ]]; then
  echo "NOTE: expected aarch64 (DGX Spark). Continuing anyway."
fi

if ! docker info >/dev/null 2>&1; then
  cat <<'EOF'

ERROR: cannot talk to the Docker daemon. Fix with:

  sudo gpasswd -a $(whoami) docker

Then log out and back in. In JupyterLab, restart the *server*, not the kernel --
group membership is inherited at login. Use $(whoami), not $USER.
EOF
  exit 1
fi
echo "Docker: OK"

# --- 2. client dependencies, in a virtual environment ------------------------
#
# DGX OS ships a PEP 668 "externally managed" Python, so installing into the
# system interpreter fails by design. A virtual environment is the correct fix.
# (--break-system-packages would also work and is a bad idea: it can break
# OS-managed packages on a machine you care about.)

say "Setting up the Python environment"

VENV="${VENV_DIR:-${SCRIPT_DIR}/.venv}"

if [[ ! -d "$VENV" ]]; then
  if ! python3 -m venv "$VENV" 2>/dev/null; then
    cat <<'EOF'

ERROR: could not create a virtual environment.

On DGX OS / Ubuntu the venv module ships separately:

    sudo apt install -y python3-venv

Then re-run ./setup.sh
EOF
    exit 1
  fi
  echo "Created $VENV"
else
  echo "Reusing $VENV"
fi

PY="$VENV/bin/python"
"$PY" -m pip install --quiet --upgrade pip
"$PY" -m pip install --quiet -r "${SCRIPT_DIR}/requirements.txt"

# Register the environment with Jupyter so demo.ipynb can select it as a kernel.
"$PY" -m ipykernel install --user \
  --name dgx-spark-demo --display-name "DGX Spark demo" >/dev/null 2>&1 \
  && echo "Registered Jupyter kernel: DGX Spark demo" \
  || echo "NOTE: could not register the Jupyter kernel (ipykernel missing?)"

echo "Done."

# --- 3. container image ------------------------------------------------------

say "Pulling the vLLM container ($IMAGE)"
docker pull "$IMAGE"

# --- 4. model weights --------------------------------------------------------

say "Model weights (~20 GB)"
echo "Model:    $MODEL"
echo "Revision: ${MODEL_REVISION:-main}"
echo

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "NOTE: HF_TOKEN is not set. Public downloads usually work, but export a"
  echo "      token if you hit rate limits:  export HF_TOKEN=hf_..."
  echo
fi

# Safe to re-run. We look in the local cache first with local_files_only=True,
# which needs no network at all: if the weights are already there, this returns
# immediately and nothing is re-downloaded. Only a cache miss reaches the Hub.
"$PY" - "$MODEL" "${MODEL_REVISION:-}" <<'PYEOF'
import sys, os
from huggingface_hub import snapshot_download
from huggingface_hub.utils import LocalEntryNotFoundError

model = sys.argv[1]
revision = sys.argv[2] or None
kwargs = {"repo_id": model}
if revision:
    kwargs["revision"] = revision

def report(path, cached):
    size = sum(os.path.getsize(os.path.join(r, f))
               for r, _, fs in os.walk(path, followlinks=True)
               for f in fs if os.path.exists(os.path.join(r, f)))
    print(f"\n  {'Already cached' if cached else 'Downloaded'}: {size/2**30:.1f} GB")
    print(f"  Path:     {path}")
    print(f"  Snapshot: {os.path.basename(path)}")

try:
    path = snapshot_download(**kwargs, local_files_only=True)
    report(path, cached=True)
    print("\n  Nothing to download. Re-running this script is free.")
except LocalEntryNotFoundError:
    print("  Not in the local cache — downloading ~20 GB across 70 files.")
    print("  Bandwidth-bound: about 3-4 minutes on a fast wired link, longer")
    print("  on slower networks. Resumable if interrupted.\n")
    path = snapshot_download(**kwargs)
    report(path, cached=False)

print("\n  Record that snapshot id alongside any benchmark numbers you publish —")
print("  it is what makes them reproducible.")
PYEOF

# --- done --------------------------------------------------------------------

cat <<EOF

$(printf '\033[1m== Setup complete\033[0m')

Next -- activate the Python virtual environment this script just built, then:

  source "${VENV}/bin/activate"

  ./serve.sh                  start the endpoint (leave running, ~4 min cold)
  jupyter lab demo.ipynb      open the walkthrough

That is the virtualenv this run created -- VENV_DIR can point it elsewhere, and
printing the literal .venv would have sent you to a different environment than
the one that has the dependencies in it.

That is a Python virtualenv, not a .env credentials file.
There is no .env in this sample and you should not create one: endpoints.json
reads NVIDIA_API_KEY from the environment, so no key is ever written to disk.

run_benchmark.py, make_chart.py and jupyter all live in .venv. If a command
reports a missing module, you have not activated it. serve.sh is the exception
-- it talks to Docker and needs no Python environment at all.

Verify the endpoint once it is up:

  curl -s http://localhost:8000/v1/models | python3 -m json.tool

EOF
