# Installation

This guide reproduces the environments used for the reference measurements. Run the commands inside the listed container using Docker, Slurm, Kubernetes, or the container runtime available at the target site.

## Common requirements

- NVIDIA GPU and driver compatible with the selected container.
- Blackwell GPU for the NVFP4 modes.
- Local FLUX checkpoints; model downloads are not performed by the benchmark runners.
- Python environments may use `--system-site-packages` to reuse CUDA-enabled PyTorch from the container.

Clone the parent samples repository:

```bash
git clone https://github.com/NVIDIA/nvidia-oci-samples.git
cd nvidia-oci-samples/generative-ai-samples/flux-1-benchmarking
make test
```

## Checkpoints

Keep checkpoints outside the repository. The runners consume these layouts:

```text
/models/
  FLUX.1-schnell-native/
    flux1-schnell.safetensors
    ae.safetensors
  google_t5-v1_1-xxl/
    pytorch_model.bin
  openai_clip-vit-large-patch14/
    model.safetensors

  FLUX.1-schnell-diffusers/
    model_index.json
    scheduler/
    transformer/
    vae/
    text_encoder/
    text_encoder_2/
    tokenizer/
    tokenizer_2/

  FLUX.1-schnell-onnx/
    clip.opt/model.onnx
    t5.opt/model.onnx
    vae.opt/model.onnx
    transformer.opt/bf16/model.onnx
    transformer.opt/fp4/model.onnx
```

Source repositories used in the experiment:

- Native and Diffusers weights: `black-forest-labs/FLUX.1-schnell`.
- TensorRT graphs: `black-forest-labs/FLUX.1-schnell-onnx`.
- Native text encoders: `google/t5-v1_1-xxl` and `openai/clip-vit-large-patch14`.

Access to BFL checkpoints may require accepting the model license and authenticating with the model registry.

## BFL PyTorch

Container:

```text
nvcr.io/nvidia/pytorch:26.07-py3
```

Install the BFL source and its runtime dependencies inside the container:

```bash
git clone https://github.com/black-forest-labs/flux.git /opt/flux
git -C /opt/flux checkout 802fb4713906133fcbd0d8dc5351620ca4773036

python3 -m venv --system-site-packages /opt/venvs/flux
source /opt/venvs/flux/bin/activate
python -m pip install --upgrade pip 'setuptools<82' wheel
python -m pip install --no-deps -e /opt/flux
python -m pip install --no-deps 'invisible-watermark==0.2.0'
python -m pip install --extra-index-url https://pypi.nvidia.com \
  accelerate einops 'fire>=0.6.0' huggingface-hub safetensors \
  sentencepiece 'transformers>=4.45.2,<5' tokenizers 'numpy==2.1.0' \
  protobuf requests colored 'opencv-python-headless==4.12.0.88' \
  'onnx>=1.18.0' 'onnxruntime~=1.22.0' onnx-graphsurgeon \
  'polygraphy>=0.49.22'
```

Verify:

```bash
PYTHONPATH=/opt/flux/src python -c 'import flux, torch; print(torch.__version__, torch.version.cuda)'
```

Run with `benchmarks.flux1_schnell.flux_batch_sweep` using `--backend pytorch`.

## Hugging Face Diffusers

Container used for the reference measurements:

```text
nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc22
```

The reference container already provided PyTorch, Diffusers, and Transformers. Verify the expected stack:

```bash
python3 - <<'PY'
import diffusers
import torch
import transformers

print('torch', torch.__version__)
print('cuda', torch.version.cuda)
print('diffusers', diffusers.__version__)
print('transformers', transformers.__version__)
PY
```

Reference versions were Diffusers `0.39.0` and Transformers `5.5.4`.

Run with `benchmark.py --mode hf-diffusers-bf16-compile` and the Diffusers checkpoint.

## TorchAO

Container:

```text
nvcr.io/nvidia/pytorch:26.07-py3
```

Create an overlay environment:

```bash
python3 -m venv --system-site-packages /opt/venvs/flux-torchao
source /opt/venvs/flux-torchao/bin/activate
python -m pip install \
  'diffusers==0.39.0' \
  'transformers==5.5.4' \
  'accelerate==1.12.0' \
  sentencepiece protobuf
```

The tested ARM64 environment additionally used this matching MSLK wheel:

```bash
python -m pip install --no-deps \
  'https://download.pytorch.org/whl/nightly/cu132/mslk-2026.8.6%2Bcu132-cp312-cp312-manylinux_2_28_aarch64.whl'
```

For other Python/CUDA/CPU architectures, install the corresponding MSLK build instead of that wheel. The measured stack reported TorchAO `0.18.0+git18278f9b` and MSLK `2026.8.6+cu132`.

Verify:

```bash
python - <<'PY'
import diffusers
import mslk
import torch
import torchao

print('torch', torch.__version__)
print('torchao', torchao.__version__)
print('mslk', mslk.__version__)
print('diffusers', diffusers.__version__)
PY
```

Run with `benchmarks.flux1_schnell.torchao_diffusers_flux_sweep` and the Diffusers checkpoint.

## SGLang

Tested official container:

```text
lmsysorg/sglang:v0.5.12
```

The image is published by the SGLang project on [Docker Hub](https://hub.docker.com/r/lmsysorg/sglang) from the [SGLang repository](https://github.com/sgl-project/sglang). The multi-platform tag digest used for validation was `sha256:42194170546745092e74cd5f81ad32a7c6e944c7111fe7bf13588152277ff356`. Verify that the selected tag contains the diffusion stack:

```bash
python3 -c 'import sglang, torch; print(sglang.__version__, torch.__version__, torch.version.cuda)'
sglang --help >/dev/null
```

If a container is not used, install the official diffusion extra in a compatible CUDA/PyTorch environment:

```bash
python3 -m pip install --upgrade pip uv
uv pip install 'sglang[diffusion]==0.5.12' --prerelease=allow
```

Run with `benchmark.py --mode sglang-bf16-offline-compile`. The runner constructs `sglang.multimodal_gen.DiffGenerator` in local mode, with no HTTP server. It submits `B` distinct requests concurrently through SGLang's local asynchronous scheduler API, uses a 100 ms dynamic-batching window, and validates that the API returns `B` decoded outputs. The configuration keeps the DiT, text encoders, and VAE on GPU, uses BF16, `torch.compile`, and `torch_sdpa` attention. Reported latency is wall time from request submission until all API outputs return; no worker hook is installed. For Nsight Systems, add `--nsys-capture` and profile one batch size at a time. The parent process then brackets the requested warmups and measured API calls with the `flux_offline_profile` NVTX range.

## vLLM Omni

Tested official container:

```text
vllm/vllm-omni:v0.26.0
```

The image is published by the vLLM Omni project on [Docker Hub](https://hub.docker.com/r/vllm/vllm-omni) from the [vLLM Omni repository](https://github.com/vllm-project/vllm-omni). The multi-platform tag digest used for validation was `sha256:5cba1538c6f8ee81e8bea6708c24e68d7b2640f466a9fbf2ef15e68f2168b48b`.

Verify the matched vLLM/vLLM Omni installation:

```bash
python3 - <<'PY'
import torch
import vllm
import vllm_omni

print('torch', torch.__version__)
print('cuda', torch.version.cuda)
print('vllm', vllm.__version__)
print('vllm_omni', vllm_omni.__version__)
PY
```

For a source installation, use Python 3.12 and keep the vLLM major/minor version aligned with vLLM Omni:

```bash
uv venv --python 3.12 --seed
source .venv/bin/activate
uv pip install 'vllm==0.26.0' --torch-backend=auto
git clone --branch v0.26.0 https://github.com/vllm-project/vllm-omni.git
uv pip install -e ./vllm-omni
```

Run with `benchmark.py --mode vllm-omni-bf16-offline`. The runner constructs `vllm_omni.entrypoints.omni.Omni`, passes `B` distinct prompt dictionaries to one `generate()` call, configures `max_num_seqs` for the largest requested batch, uses a 100 ms request-batching window, and validates that the API returns `B` images. Reported latency is wall time for `generate()` to return all outputs; no worker hook is installed. Model loading and warmup compilation remain outside measured time. For Nsight Systems, add `--nsys-capture` and profile one batch size at a time. The parent process then brackets the requested warmups and measured API calls with the `flux_offline_profile` NVTX range.

## TensorRT-LLM VisualGen

Container:

```text
nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc22
```

No additional overlay was needed for the measured VisualGen path. Verify:

```bash
python3 - <<'PY'
import tensorrt_llm
from tensorrt_llm import VisualGen, VisualGenArgs

print('tensorrt_llm', tensorrt_llm.__version__)
print('VisualGen', VisualGen)
print('VisualGenArgs', VisualGenArgs)
PY
```

Run with `benchmarks.flux1_schnell.visualgen_flux_sweep`, the Diffusers checkpoint, and one YAML file from `benchmarks/flux1_schnell/configs/`.

## TensorRT

Container:

```text
nvcr.io/nvidia/pytorch:26.07-py3
```

Use the BFL installation described above. The tested container provided TensorRT `11.1.0.106`; Polygraphy and the ONNX packages are installed by the BFL setup commands.

Verify:

```bash
PYTHONPATH=/opt/flux/src python - <<'PY'
import flux
import polygraphy
import tensorrt

print('tensorrt', tensorrt.__version__)
print('polygraphy', polygraphy.__version__)
PY
```

Build one static plan set per batch size:

```bash
PYTHONPATH="$PWD:/opt/flux/src" python3 -m benchmarks.flux1_schnell.flux_batch_sweep \
  --backend trt \
  --precision bf16 \
  --variant eager \
  --batch-sizes 1 \
  --engine-root /engines/flux-schnell-1024 \
  --onnx-dir /models/FLUX.1-schnell-onnx \
  --output-dir results/tensorrt-build \
  --build-only
```

Repeat for each required batch size and for `--precision fp4` when using the explicit-NVFP4 ONNX graph. The reference stack built NVFP4 only at B1; B2+ plan compilation was unavailable.

## Modes

See [`ENVIRONMENTS.md`](ENVIRONMENTS.md) for the complete runner, mode, precision, checkpoint, and reference-status lookup table.
