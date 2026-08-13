# FLUX.1-schnell inference on one GB300 — fair comparison summary

*Workload and measurement contract*

• Model: `black-forest-labs/FLUX.1-schnell`, approximately 12B transformer parameters.
• Hardware: one GB300 GPU.
• Image workload: `1024×1024`, four denoising steps, guidance `0.0`.
• True request batching: batch `B` contains `B` distinct prompt entries, produces `B` latent outputs, and VAE-decodes `B` images. This is not one prompt embedding repeated `B` times.
• Every successful point uses two warmups and 20 measured iterations with the same canonical prompt bank and deterministic seed configuration.
• Internal-runtime timing starts before T5/CLIP text encoding and ends after the complete decoded `B`-image tensor is available. PyTorch, Diffusers, and TensorRT explicitly synchronize CUDA; VisualGen uses its synchronous blocking request path. Offline SGLang and vLLM Omni instead measure their public in-process API until all outputs return, including local scheduling and batching overhead.
• `images/s = B / batch latency` within each stated timing scope. Offline API throughput is reported separately from synchronized internal-runtime throughput.
• Excluded from timing: model/checkpoint loading, engine building, warmup, PIL/PNG conversion, GPU-to-host image copy, file I/O, and network serving.
• All successful points passed prompt-count, prompt-digest, output-count, iteration-count, and timing-scope contract checks. Engine-internal fusion is not asserted by the offline API result contract. These checks do not validate image quality or numerical accuracy.

*Checkpoint provenance and execution precision*

• Native BF16 checkpoint: `black-forest-labs/FLUX.1-schnell`, revision `741f7c3ce8b383c54771c7003378a50191e9efe9`. Used by BFL PyTorch; source and execution are BF16.
• Diffusers BF16 checkpoint: the same BFL repository/revision in Diffusers layout. Used by HF Diffusers, TorchAO, SGLang, vLLM Omni, and TRT-LLM VisualGen.
• BFL ONNX checkpoint: `black-forest-labs/FLUX.1-schnell-onnx`, revision `7ad1eace4e708f71d82902ce08389444134cdf0d`. It contains explicit BF16 and NVFP4 transformer graphs plus BF16 T5, CLIP, and VAE graphs, and is used only by TensorRT.
• TensorRT plans were built separately for each static batch size and `1024×1024` shape.
  • BF16 plans were built through B32. The explicit-NVFP4 ONNX graph builds/runs only at B1; B2+ fails in TensorRT/Myelin compilation.
• For every NVFP4 option, only the FLUX transformer is quantized. T5, CLIP, and VAE remain BF16.

*How each path runs*

• BFL PyTorch: native BFL code and BF16 safetensors; `torch.compile` is the best BFL configuration.
• HF Diffusers BF16: Diffusers pipeline with BF16 weights/activations; full transformer compile is the best standard Diffusers configuration.
• TorchAO NVFP4: starts from the BF16 Diffusers checkpoint, selectively converts large transformer linear layers to NVFP4 weights with dynamically scaled NVFP4 activations, and executes in PyTorch with regional compilation. No ONNX export or TensorRT conversion is involved.
• SGLang: loads the BF16 Diffusers checkpoint through `DiffGenerator` in local mode. The runner submits concurrent requests directly to the local asynchronous scheduler API; no HTTP server is started. The measured configuration uses a 100 ms dynamic-batching window, `torch.compile`, GPU-resident components, and `torch_sdpa` attention.
• vLLM Omni: loads the BF16 Diffusers checkpoint through the offline `Omni` API. One `generate()` call receives all `B` prompt dictionaries, with `max_num_seqs=32` and a 100 ms request-batching window. The GB300 run selects `CUDNN_ATTN` and regional `torch.compile`; no HTTP server is started.
• TRT-LLM VisualGen: loads the BF16 Diffusers checkpoint through the VisualGen PyTorch backend. Its NVFP4 path uses dynamic NVFP4 custom ops; it is not the BFL NVFP4 ONNX/TensorRT engine.
• TensorRT: loads the BFL ONNX graphs and builds static TensorRT plans. BF16 uses BF16 transformer weights/activations; NVFP4 uses the explicit BFL NVFP4 transformer graph.

*Fair end-to-end throughput: best measured configuration for each path*

```text
Images/s, BF16 transformer; T5/CLIP/VAE are BF16

Execution path                    Checkpoint → runtime W/A             B1    B2    B4    B8   B16   B32
BFL PyTorch compile               BF16 native → BF16/BF16            3.19  3.72  3.99  4.10  4.19  4.21
HF Diffusers compile              BF16 Diffusers → BF16/BF16         3.26  3.58  3.98  4.14  4.22  4.51
HF/TorchAO regional compile       BF16 Diffusers → BF16/BF16         3.22  3.52  3.84  4.03  4.12  4.41
TRT-LLM VisualGen best per B      BF16 Diffusers → BF16/BF16         3.34  3.61  3.86  4.05  4.11  4.34
TensorRT best per B               BF16 ONNX → BF16/BF16              4.05  4.03  4.12  4.13  4.25  4.31
```

For the two `best per B` rows, the better measured eager-launch or CUDA-Graph result is selected independently at each batch. VisualGen uses CUDA Graph through B16 and eager launch at B32; TensorRT uses CUDA Graph at B1 and eager launch at B2+.

The offline engines use a broader API-completion timing scope, so they are not ranked in the synchronized internal-runtime table above:

```text
Images/s, BF16 offline API completion (includes local scheduling and batching)

Execution path                    Checkpoint → runtime W/A             B1    B2    B4    B8   B16   B32
SGLang offline batch + compile    BF16 Diffusers → BF16/BF16         1.86  2.27  2.56  2.58  2.33  1.95
vLLM Omni offline batch           BF16 Diffusers → BF16/BF16         2.43  2.71  2.99  3.17  3.22  3.43
```

```text
Images/s, NVFP4 transformer; T5/CLIP/VAE remain BF16

Execution path                    Checkpoint → runtime W/A             B1    B2    B4    B8   B16   B32
TorchAO regional best per B       BF16 Diffusers → NVFP4/NVFP4       3.84  3.89  4.41  4.69  4.80  5.23
TRT-LLM VisualGen best per B      BF16 Diffusers → NVFP4/NVFP4       3.69  3.61  4.00  4.24  4.34  4.68
TensorRT NVFP4 + CUDA Graph       BFL NVFP4 ONNX → NVFP4/NVFP4      6.71  N/A   N/A   N/A   N/A   N/A
```

TorchAO uses CUDA Graph at B1–B2 and regional compile without CUDA Graph at B4+. VisualGen selects CUDA Graph at B1–B4 and eager launch at B8+. TensorRT NVFP4 B2+ is `N/A` because no valid plan could be built, not because those batches were measured as slow.

*Main observations*

• Best B1 result: TensorRT NVFP4 + CUDA Graph, `6.71 images/s`, `149.1 ms` synchronized batch latency.
• Best scalable NVFP4 path: TorchAO regional compile. It reaches `4.41/4.69/4.80/5.23 images/s` at B4/B8/B16/B32 and outperforms VisualGen NVFP4 at every measured batch. At B1, TensorRT NVFP4 is faster than both Torch paths.
• Best scalable BF16 paths: TensorRT and HF Diffusers compile are close around B4–B16. HF Diffusers compile has the highest B32 BF16 result at `4.51 images/s`; TensorRT has the best B1 BF16 latency at approximately `247 ms`.
• Offline API completion: SGLang reaches `1.86/2.27/2.56/2.58/2.33/1.95 images/s` from B1 through B32; vLLM Omni reaches `2.43/2.71/2.99/3.17/3.22/3.43 images/s`. Every measured call returned all requested outputs. These broader API timings are not directly comparable to the synchronized internal-runtime rows.
• VisualGen provides functional dynamic NVFP4 batching through B32, but TorchAO is faster for scalable NVFP4 in this experiment.
• For fixed B1 production inference, prioritize the BFL TensorRT NVFP4 ONNX path. For batching, fine-tuning compatibility, or a pure-Torch deployment, prioritize TorchAO NVFP4 regional compile: CUDA Graph for B1–B2 and non-CUDA-Graph regional compile for B4+.

*Limitations and interpretation*

• These results establish performance, not quantized image quality. LPIPS/FID plus representative-prompt review are still required.
• Only `1024×1024` square generation was benchmarked here. Other resolutions and aspect ratios need separate plans and measurements.
• This is model-pipeline end-to-end latency, not full service latency. Request queuing, network transport, image serialization, post-processing, super-resolution, and stitching are outside the timer.

*Reproducibility artifacts*

• This repository contains the benchmark runners, prompt bank, fair-measurement validator, and Slurm example.
• Raw checkpoints, generated TensorRT plans, per-run JSON, images, and Nsight reports are intentionally kept outside Git.
• Nsight coverage: TensorRT, TorchAO, and TRT-LLM VisualGen at B1/B4 where supported. CUDA Graph profiles use `--cuda-graph-trace=node`. Offline SGLang and vLLM Omni use a parent-process NVTX range around the requested warmup and measured API calls; they do not inject worker hooks. TensorRT NVFP4 B4 is unavailable because no valid B4 plan exists.
