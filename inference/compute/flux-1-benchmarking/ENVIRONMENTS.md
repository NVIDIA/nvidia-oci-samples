# Runtime environments and mode lookup

## What hardware is needed

- Reference measurements used one NVIDIA GB300 GPU (`sm_103`) on an ARM64 Slurm node.
- BF16 modes can run on other CUDA GPUs with sufficient memory, but their numbers are not directly comparable to the reference report.
- NVFP4 modes require Blackwell plus compatible TorchAO, TensorRT, or TensorRT-LLM builds.
- The Diffusers BF16 B1 smoke run reserved about 36.3 GiB. A 48 GiB or larger GPU is a practical minimum for B1; larger batches can require more memory.
- Slurm is optional. The Python runners work directly in the same environments; the checked-in Slurm script supports native execution or a configured container image.

## Containers used for the reference measurements

These images produced the reference measurements. The final image was used only for an additional development probe.

| Role | Container image |
|---|---|
| BFL PyTorch, TensorRT, TorchAO | `nvcr.io/nvidia/pytorch:26.07-py3` |
| HF Diffusers, TRT-LLM VisualGen | `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc22` |
| SGLang | `lmsysorg/sglang:v0.5.12` |
| vLLM Omni | `vllm/vllm-omni:v0.26.0` |
| Additional TensorRT development probe only | `nvcr.io/nvidia/cuda:13.3.1-tensorrt-devel-ubuntu24.04` |

Use these image references with the container runtime available in the target environment.

## Observed software stacks

| Environment | Observed versions and overlays |
|---|---|
| PyTorch 26.07 base | PyTorch `2.13.0a0+9186a08b2c.nv26.07`, CUDA `13.3`, TensorRT `11.1.0.106` |
| BFL native/TensorRT overlay | BFL FLUX source commit `802fb4713906133fcbd0d8dc5351620ca4773036` plus Transformers `<5`, ONNX, Polygraphy, and TensorRT dependencies |
| TorchAO overlay | TorchAO `0.18.0+git18278f9b`, MSLK `2026.8.6+cu132`, Diffusers `0.39.0`, Transformers `5.5.4` |
| TensorRT-LLM 1.3.0rc22 base | PyTorch `2.12.0a0+5aff3928d8.nv26.05`, CUDA `13.2`, TensorRT-LLM `1.3.0rc22` |
| Diffusers in TRT-LLM base | Diffusers `0.39.0`, Transformers `5.5.4` |
| SGLang official ARM64 image | SGLang `0.5.12`, PyTorch `2.11.0+cu130`, CUDA `13.0`; dynamic diffusion request batching with the tested `torch_sdpa` attention backend |
| vLLM Omni official ARM64 image | vLLM Omni `0.26.0`, PyTorch `2.11.0+cu130`, CUDA `13.0`; FLUX request batching with `CUDNN_ATTN` selected on GB300 |
| TensorRT 11.2 probe overlay | `tensorrt-cu13==11.2.1.2` installed into a venv on the PyTorch 26.07 base; not used for the reference table |

## Mode lookup

`W/A` describes transformer weight/activation precision. T5, CLIP, and VAE remain BF16 for every NVFP4 row.

| Execution path | Unified `--mode` | Transformer W/A | Source checkpoint | Reference status |
|---|---|---|---|---|
| BFL PyTorch compile | `pytorch-bf16-compile` | BF16/BF16 | BFL native BF16 | Measured |
| HF Diffusers compile | `hf-diffusers-bf16-compile` | BF16/BF16 | Diffusers BF16 | Measured |
| TorchAO BF16 regional compile | `torchao-diffusers-bf16-regional` | BF16/BF16 | Diffusers BF16 | Measured |
| TorchAO NVFP4 regional compile | `torchao-diffusers-nvfp4-regional` | NVFP4/NVFP4 | Diffusers BF16 | Measured |
| TorchAO NVFP4 regional + CUDA Graph | `torchao-diffusers-nvfp4-regional-cg` | NVFP4/NVFP4 | Diffusers BF16 | Measured |
| SGLang offline batch + compile | `sglang-bf16-offline-compile` | BF16/BF16 | Diffusers BF16 | B1–B32 measured |
| vLLM Omni offline batch | `vllm-omni-bf16-offline` | BF16/BF16 | Diffusers BF16 | B1–B32 measured |
| VisualGen BF16 | `trtllm-visualgen-bf16` | BF16/BF16 | Diffusers BF16 | Measured |
| VisualGen BF16 + CUDA Graph | `trtllm-visualgen-bf16-cuda-graph` | BF16/BF16 | Diffusers BF16 | Measured |
| VisualGen dynamic NVFP4 | `trtllm-visualgen-nvfp4` | NVFP4/NVFP4 | Diffusers BF16 | Measured |
| VisualGen dynamic NVFP4 + CUDA Graph | `trtllm-visualgen-nvfp4-cuda-graph` | NVFP4/NVFP4 | Diffusers BF16 | Measured |
| TensorRT BF16 | `trt-bf16-eager` | BF16/BF16 | BFL BF16 ONNX | Measured |
| TensorRT BF16 + CUDA Graph | `trt-bf16-cuda-graph` | BF16/BF16 | BFL BF16 ONNX | Measured |
| TensorRT NVFP4 | `trt-fp4-eager` | NVFP4/NVFP4 | BFL NVFP4 ONNX | B1 measured; B2+ plan unavailable |
| TensorRT NVFP4 + CUDA Graph | `trt-fp4-cuda-graph` | NVFP4/NVFP4 | BFL NVFP4 ONNX | B1 measured; B2+ plan unavailable |

SGLang and vLLM Omni report public offline-API completion time, including local scheduling and batching. Their results are kept separate from synchronized internal-runtime measurements.

Run these values through `python3 benchmark.py --mode <mode>`. Advanced runner mapping:

```text
BFL PyTorch and TensorRT  benchmarks.flux1_schnell.flux_batch_sweep
HF Diffusers              benchmarks.flux1_schnell.hf_diffusers_flux_sweep
TorchAO                   benchmarks.flux1_schnell.torchao_diffusers_flux_sweep
TensorRT-LLM VisualGen    benchmarks.flux1_schnell.visualgen_flux_sweep
SGLang                    benchmarks.flux1_schnell.sglang_flux_sweep
vLLM Omni                 benchmarks.flux1_schnell.vllm_omni_flux_sweep
```
