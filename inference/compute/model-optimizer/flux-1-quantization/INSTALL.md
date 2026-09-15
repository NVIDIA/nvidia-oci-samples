# Install

Two environments. Quantization needs Model Optimizer; serving needs TensorRT-LLM.

**They cannot share one container, and the reason is worth knowing before you try.** The TensorRT-LLM release container ships `nvidia-modelopt 0.37.0`, which has no Diffusers-format export — the `--hf-ckpt-dir` flag does not exist, so the export stage cannot produce the artefact at all. **0.42.0 is the floor.** But TensorRT-LLM 1.3.0rc22 pins `nvidia-modelopt~=0.37.0` and `onnx>=1.21`, while modelopt 0.42.0 pins `onnx~=1.19`. Those constraints do not intersect: installing what the export needs breaks TensorRT-LLM in the same container. Quantize in one environment, serve in another.

Run everything from a workspace with enough disk. See the workspace section in [README.md](README.md).

## Model Optimizer overlay

For the `export` stage.

```bash
# Container
nvcr.io/nvidia/pytorch:26.07-py3

# The public Model Optimizer repository. The example script lives only in the
# repo, and its tag must match the installed library.
git clone https://github.com/NVIDIA/Model-Optimizer.git "$FLUX_QUANT_WORKSPACE/src/Model-Optimizer"
export MODELOPT_DIFFUSERS_DIR="$FLUX_QUANT_WORKSPACE/src/Model-Optimizer/examples/diffusers/quantization"

# Pinned deliberately. 0.42.0 is the floor -- earlier releases have no
# Diffusers-format export, and the failure does not surface until the export
# stage, hours in.
python3 -m pip install 'nvidia-modelopt[hf,onnx]>=0.42.0'
python3 -m pip install -r "$FLUX_QUANT_WORKSPACE/src/Model-Optimizer/examples/diffusers/requirements.txt"
```

Verify:

```bash
python3 -c 'import modelopt; print(modelopt.__version__)'
ls "$MODELOPT_DIFFUSERS_DIR/quantize.py"
```

The harness finds the example through `MODELOPT_DIFFUSERS_DIR`, then `$WORKSPACE/src/Model-Optimizer/...`, then a couple of conventional locations. Set the variable and you can ignore the search order.

## TensorRT-LLM

For the `dynamic` and `verify` stages.

```bash
# Container
nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc22
```

Diffusers and Transformers are already present. Verify:

```bash
python3 -c 'import torch, diffusers; print(torch.__version__, diffusers.__version__)'
python3 -c 'import tensorrt_llm; print(tensorrt_llm.__version__)'
```

The shipped FLUX fp4 config lives at `examples/visual_gen/configs/flux1-dev-fp4-1gpu.yaml` inside the container. It uses **dynamic** quantization. Loading a statically quantized checkpoint is the thing the `verify` stage is testing, so do not assume it works — if it does not, that is a finding to record rather than a bug to route around.

## TorchAO

Only if you run the `dynamic` stage through Diffusers rather than VisualGen.

```bash
python3 -m pip install torchao
python3 -c 'from torchao.prototype.mx_formats import NVFP4InferenceConfig; print("ok")'
```

## Stages that need no container at all

`preflight`, `schema` and `quality` need no GPU and no container. Useful when an
allocation has ended and you only want to re-check something on a login node.

A virtual environment is the least painful route, because login nodes are
usually PEP 668 "externally managed" and refuse a plain `pip install`. Put it on
a persistent volume rather than in a home directory, which is often only a few
gigabytes:

```bash
python3 -m venv <persistent-path>/venv-cpu
source <persistent-path>/venv-cpu/bin/activate

# CPU-only torch -- ~180 MB, against several GB for the CUDA build
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install safetensors huggingface_hub
```

Without a venv, `pip` needs `--break-system-packages` on any PEP 668 system.

### The `schema` stage's authoritative check needs the full chain

`schema` runs two checks. The **name-based** one works with the packages above
and only compares module names against a documented list. The **authoritative**
one imports Model Optimizer's real `filter_func_flux_dev` and runs it over every
module in the export — and that import drags in the example's whole dependency
tree, five levels deep:

```bash
pip install diffusers datasets "nvidia-modelopt==0.42.0" onnx onnxruntime onnxslim
```

Note `<Model-Optimizer>/examples/diffusers/requirements.txt` does **not** cover
this — it lists only `nvtx`, `opencv-python` and `sentencepiece`.

Without the chain the stage still completes and reports dtypes, but prints
`exclusion filter not checked` with the specific import that failed. Read that
line. `filter_agrees` is the check that actually proves the right layers were
protected; a run without it looks almost identical and proves much less.

A successful run says:

```text
exclusion filter: utils.py:filter_func_flux_dev in <checkout>
VERIFIED: every layer the filter excludes is at high precision (656 modules checked)
```

The module count matters. If it is far below the weight-bearing module total,
the filter raised on the rest and the result is inconclusive rather than passing.

**Both of these catch people out, and neither is obvious.**

`safetensors` is never installed on its own — it arrives as a dependency of
`transformers`, so it is present inside either container and absent on a bare
login node.

`torch` is needed by `schema` even though the stage does no tensor maths: it
imports Model Optimizer's `filter_func_flux_dev` from source, and that import
pulls torch in. Without it the stage fails with `No module named 'torch'`.

Both failures land *after* you have set up everything else, one at a time.

## Metrics

For the `quality` stage. These need no special container, but CLIP scoring uses
`torch`, so a GPU is used when one is present.

```bash
python3 -m pip install pillow numpy transformers torch

# CMMD. Optional but recommended: it is the primary distributional metric here.
# Note this is a script repository, not a package -- there is no setup.py, so
# `pip install` fails. Clone it and point CMMD_REPO at the checkout.
git clone https://github.com/sayakpaul/cmmd-pytorch.git /path/to/cmmd-pytorch
export CMMD_REPO=/path/to/cmmd-pytorch
```

The harness also looks in `$WORKSPACE/src/cmmd-pytorch` and a couple of conventional locations, so setting the variable is optional if you clone it there.

If CMMD is unavailable the stage still runs, reports PSNR and CLIP, and prints why CMMD was skipped. It never substitutes a different metric silently.

## Hugging Face access

FLUX.1 checkpoints are gated. Both steps are required, and this is the most common first failure.

1. Accept the licence in a browser, once per model:
   - https://huggingface.co/black-forest-labs/FLUX.1-dev
   - https://huggingface.co/black-forest-labs/FLUX.1-schnell
2. Authenticate:

```bash
hf auth login                  # or: export HF_TOKEN=hf_...
```

Point the cache at the workspace, since a 34 GB model will not fit in a home directory:

```bash
export HF_HOME="$FLUX_QUANT_WORKSPACE/hf"
```

`python3 quantize.py --stage preflight` checks the token and confirms each gated repository is actually readable, rather than waiting for the download to fail.

## Confirming the environment

```bash
make preflight
```

Preflight fails loudly on anything that would stop a later stage: no GPU, insufficient disk, missing `torch`, no Hugging Face token, an unaccepted licence. It also warns when the workspace looks node-local, which matters because node-local storage is normally wiped when the allocation ends.

## Inside the container, on a shared cluster

Container runtimes normally mount your home directory and inherit the environment of the shell that started them. Both are convenient and both cause failures whose error messages point somewhere other than the cause. Set these before running anything:

```bash
# Ignore ~/.local/lib/pythonX.Y/site-packages
export PYTHONNOUSERSITE=1

# Drop the scheduler variables the container inherited
unset $(env | awk -F= '/^(SLURM_|PMI_|PMIX_)/ {print $1}')
```

**Why the first one.** Python puts `~/.local/lib/pythonX.Y/site-packages` ahead of the container's own packages. If you have ever run `pip install --user torch` on the login node, that copy wins, and its compiled extension does not match the rest of the container:

```text
ImportError: cannot import name '_is_kineto_stopped' from 'torch._C._autograd'
  (/home/<user>/.local/lib/python3.12/site-packages/torch/_C.cpython-312-...so)
```

The path in the message is the diagnosis. Confirm you are on the container's build with `python3 -c 'import torch; print(torch.__file__)'` — you want a path under `/usr/local`, not one under your home directory.

**Why the second one.** Open MPI checks for scheduler variables at import time. Finding them, it concludes it was launched under `srun` and reaches for a PMI it was not built against, so `import tensorrt_llm` aborts the process:

```text
OPAL ERROR: Unreachable in file pmix3x_client.c at line 111
The application appears to have been direct launched using "srun",
but OMPI was not built with SLURM's PMI support
*** An error occurred in MPI_Init_thread
```

Nothing is wrong with the installation. The container simply inherited variables describing a job it is not part of.

**Environment does not survive shell transitions.** A new `tmux` session, a nested `bash`, and the container itself each start clean. Export `FLUX_QUANT_WORKSPACE`, `HF_HOME`, `CMMD_REPO` and the two settings above *inside* the shell that will run the stages, not in the one you launched it from.

**Node-local scratch is not always where you expect.** Container runtimes want a writable node-local path for their unpacked image, and the conventional one is not writable on every node. Find one before you start:

```bash
for d in /raid /scratch/local /scratch.local /local /tmp; do
  [ -d "$d" ] || continue
  if touch "$d/.wtest.$$" 2>/dev/null; then
    rm -f "$d/.wtest.$$"
    echo "WRITABLE  $d   $(df -h "$d" | awk 'NR==2{print $4}') free"
  fi
done
```

Unpack the image node-local rather than onto shared scratch. It is several hundred thousand small files, and every import walks them.
