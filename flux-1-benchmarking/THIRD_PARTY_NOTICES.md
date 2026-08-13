# Third-party notices

This directory contains NVIDIA-authored Apache-2.0 benchmark code and limited adaptations of the projects identified below. It is intended to be distributed as `generative-ai-samples/flux-1-benchmarking` within [`NVIDIA/nvidia-oci-samples`](https://github.com/NVIDIA/nvidia-oci-samples), under that repository's top-level license. It does not redistribute model checkpoints, compiled inference engines, Python environments, or container images.

## Adapted source

### Black Forest Labs FLUX

- Project: [black-forest-labs/flux](https://github.com/black-forest-labs/flux)
- Source revision: [`802fb4713906133fcbd0d8dc5351620ca4773036`](https://github.com/black-forest-labs/flux/tree/802fb4713906133fcbd0d8dc5351620ca4773036)
- Files adapted: `benchmarks/flux1_schnell/flux_batch_sweep.py` and `benchmarks/flux1_schnell/flux_t2i_trt11.py`
- Upstream source areas: `src/flux/cli.py`, `src/flux/trt/engine/base_engine.py`, `src/flux/trt/trt_config/base_trt_config.py`, and `src/flux/trt/trt_manager.py`
- License: Apache License 2.0; see [`third_party/licenses/Apache-2.0.txt`](third_party/licenses/Apache-2.0.txt)
- Copyright and authorship: The adapted TensorRT integration areas identify Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved. The upstream FLUX project identifies Black Forest Labs as the project author.

### TorchAO

- Project: [pytorch/ao](https://github.com/pytorch/ao)
- Source revision: [`4ee58b57d0fbed12a88abfa86440cd375b9890e1`](https://github.com/pytorch/ao/tree/4ee58b57d0fbed12a88abfa86440cd375b9890e1)
- File adapted: `benchmarks/flux1_schnell/torchao_diffusers_flux_sweep.py`
- Upstream source: `benchmarks/quantization/eval_accuracy_and_perf_of_flux.py`
- License: BSD 3-Clause; see [`third_party/licenses/BSD-3-Clause.txt`](third_party/licenses/BSD-3-Clause.txt)
- Copyright: Copyright 2023 Meta. All contributions by Arm: Copyright (c) 2024-2026 Arm Limited and/or its affiliates. The adapted upstream source file also identifies Copyright (c) Meta Platforms, Inc. and affiliates.

Modifications to these files are Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. Source headers identify the applicable upstream license and the parent repository's Apache-2.0 license.

## Separately acquired runtime dependencies

The benchmark can interoperate with the following projects. They are not included in this repository and remain subject to their own licenses and notices.

| Project | Use | License or terms |
|---|---|---|
| [Black Forest Labs FLUX](https://github.com/black-forest-labs/flux) | Native FLUX implementation and TensorRT integration | Apache License 2.0 |
| [Hugging Face Diffusers](https://github.com/huggingface/diffusers) | Diffusers FLUX pipeline | Apache License 2.0 |
| [Hugging Face Transformers](https://github.com/huggingface/transformers) | Text encoders and tokenizers | Apache License 2.0 |
| [PyTorch](https://github.com/pytorch/pytorch) | Tensor execution and compilation | BSD-style license |
| [TorchAO](https://github.com/pytorch/ao) | Quantization | BSD 3-Clause |
| [MSLK](https://pypi.org/project/mslk-cuda/) | Kernels used by the tested TorchAO environment | BSD 3-Clause |
| [SGLang](https://github.com/sgl-project/sglang) | Offline diffusion runtime | Apache License 2.0 |
| [vLLM Omni](https://github.com/vllm-project/vllm-omni) | Offline diffusion runtime | Apache License 2.0 |
| [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) | VisualGen runtime | Apache License 2.0 |
| [NumPy](https://github.com/numpy/numpy) | Array conversion | BSD 3-Clause |
| [Pillow](https://github.com/python-pillow/Pillow) | Image conversion | HPND license |
| [ONNX](https://github.com/onnx/onnx), [ONNX Runtime](https://github.com/microsoft/onnxruntime), and [Polygraphy](https://github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy) | TensorRT graph tooling | Apache License 2.0 or MIT, as identified by each project |
| NVIDIA CUDA, TensorRT, and NGC containers | GPU runtime and tested environments | Applicable NVIDIA product and container terms |

Checkpoint names in the documentation are identifiers only. No checkpoint or model license is granted by this repository. Obtain each checkpoint from its publisher and review its license or access terms before use.

If a container, virtual environment, checkpoint, engine, or other binary artifact is redistributed with this project, its complete reviewed dependency inventory, license texts, notices, and applicable product terms must be added for that exact artifact before distribution.
