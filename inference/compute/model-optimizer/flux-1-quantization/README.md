# FLUX.1 NVFP4 quantization

**Produce a static NVFP4 FLUX.1 checkpoint using only the public [NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer), and show its output quality holds.**

The goal is a recipe you can run on your own model, using nothing but the public repository. Where the public path falls short, that gap is recorded here rather than patched around locally — because the gap is what anyone adopting this will hit too.

**This is deliberately the static path.** Runtime, or *dynamic*, quantization compresses the model as it loads and produces excellent results, but it involves no Model Optimizer and no checkpoint, so there is nothing to hand anyone. It is a flag, not a recipe. Static post-training quantization is the thing that transfers.

Everything here uses public tooling on purpose. If a step does not work with the public repository, that gap is the finding, and it belongs upstream rather than being patched around locally.

> **Read [INSTALL.md](INSTALL.md) before the first run.** Quantization and serving need separate containers, and the reason is a dependency conflict that is not obvious until it wastes an allocation: the Model Optimizer shipped in the TensorRT-LLM container is too old to produce a Diffusers export at all, and upgrading it breaks TensorRT-LLM.

## Quick start

```bash
python3 quantize.py --list-stages
python3 quantize.py --stage preflight
```

Preflight takes seconds and catches the failures that otherwise waste an allocation. Run it before anything else.

**Then run the stages in two groups, because they need different environments.** A single `--from download --through quality` spans the container boundary described above and will fail partway:

```bash
# Model Optimizer environment
python3 quantize.py --stage download
python3 quantize.py --stage export

# TensorRT-LLM environment
python3 quantize.py --stage dynamic
python3 quantize.py --stage verify

# no GPU, no container -- see INSTALL.md
python3 quantize.py --stage schema
python3 quantize.py --stage quality
```

`--dry-run` prints what each stage would do and records nothing, so it is safe to run at any point.

## Two artefacts, two consumers

`export` writes both, and the [Model Optimizer diffusers README](https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/diffusers#important-parameters) sends them to different places. Getting this wrong costs a day:

| Artefact | Consumer |
|---|---|
| `exports/<model>/torch/backbone.pt` | **PyTorch**, restored with `modelopt.torch.opt.restore()` |
| `exports/<model>/hf/` | **SGLang, vLLM, TRT-LLM** |

A stock `FluxPipeline.from_pretrained()` on the Hugging Face export is on neither list, and fails. The directory has exactly the shape of a normal Diffusers model, which makes the mistake easy and the error message unhelpful.

Note the PyTorch checkpoint is the same size as the BF16 transformer. It stores the original weights plus quantizer modules and their calibrated scales, and simulates the quantization in the forward pass. That reproduces NVFP4 numerics faithfully — which is what a quality question needs — but runs through BF16 storage, so it says nothing about speed.

## Stages

| Stage | GPU | What it does |
|---|---|---|
| `preflight` | no | GPU, disk, packages, Hugging Face access. Writes `environment.json` |
| `download` | no | BF16 baseline and the published NVFP4 reference, with revisions pinned |
| `dynamic` | yes | Preliminary quality read using runtime NVFP4. No calibration, no checkpoint |
| `export` | yes | **Static PTQ with public Model Optimizer.** Writes a PyTorch checkpoint and a Hugging Face export |
| `verify` | yes | **Restores the PyTorch checkpoint with `mto.restore` and generates from it** |
| `schema` | no | Compares our export against the published checkpoint |
| `quality` | no | Scores paired images: CMMD, PSNR, CLIP |

**`preflight`, `schema` and `quality` need no GPU and no container.** They read files, import the exclusion filter from source, and score images — useful once an allocation has ended. On a login node they need a small virtual environment first; see [INSTALL.md](INSTALL.md#stages-that-need-no-container-at-all).

**`export` then `verify` is the deliverable.** Static PTQ produces the checkpoint, and `verify` restores it with `mto.restore` and generates from it so the quality can be scored. That pair is the recipe someone would repeat on a model of their own.

**`dynamic` is a preliminary check, not the goal.** It needs no calibration and no checkpoint, so it answers "does NVFP4 arithmetic hurt this model" in an afternoon — worth doing first, and cheap. But it uses torchao rather than Model Optimizer and produces nothing transferable, so a good dynamic result is encouraging rather than sufficient.

The two arms also differ in what they touch, which matters when comparing them:

| | Layers quantized | Attention |
|---|---|---|
| `dynamic` | Model Optimizer's exclusions applied via `filter_fn` | BF16 |
| `export` | Same exclusions, from `filter_func_flux_dev` | FP8 quantizers written into the checkpoint by `--quantize-mha` |

Note the attention row carefully. `--quantize-mha` puts FP8 quantizers *in the
checkpoint*; whether a given serving backend executes them, and at what
accumulator precision, is decided by the runtime rather than by the export. Do
not assume the two arms differ only in linear-layer precision when comparing
them.

Without the filter, torchao converts *every* linear including the ten the recipe protects. That is a more aggressive recipe than anything NVIDIA ships, and it measured 4.8× further from BF16 than the static export. `--no-exclusions` reproduces that deliberately if you want to quantify what the exclusions buy.

## Serving the checkpoint

The stages above end at a scored checkpoint. Deploying it is a separate phase with two `make` targets:

```bash
make bench             # throughput: BF16, dynamic and static, per model
make serve-quality     # quality scored from the SERVED checkpoint
```

Both need the **TensorRT-LLM container**, not the quantization one — see [INSTALL.md](INSTALL.md). Both run a preflight first and refuse to start if an input is missing, because a partial run still writes numbers and a table missing a row looks like a result.

This phase exists because `mto.restore` cannot answer a throughput question. It reproduces NVFP4 numerics on BF16 storage, so it is a fair instrument for quality and tells you nothing at all about speed. Only the packed `exports/<model>/hf/` export on real NVFP4 kernels can.

### One upstream bug will stop you before anything else

TensorRT-LLM VisualGen cannot load *any* Model Optimizer FLUX export as shipped. The model never finishes building:

```text
TypeError: unsupported operand type(s) for *: 'int' and 'NoneType'
```

From `tensorrt_llm/_torch/visual_gen/models/flux/transformer_flux.py`:

```python
out_channels = getattr(pretrained_config, "out_channels", in_channels)   # intended fallback
```

`getattr`'s third argument only applies when the attribute is **missing**. Every FLUX config ships `"out_channels": null` — the attribute is there, set to `None` — so the fallback never fires and `None` reaches a multiplication. Diffusers has always written `out_channels or in_channels`, which covers both cases, which is why the null causes no trouble anywhere else.

The fix is one line:

```python
- out_channels = getattr(pretrained_config, "out_channels", in_channels)
+ out_channels = getattr(pretrained_config, "out_channels", None) or in_channels
```

**You do not have to wait for that to land upstream.** The `export` stage writes `out_channels` into the exported `transformer/config.json`, keeps the untouched original as `config.json.orig`, and records the edit in the run manifest under `post_export_edits`. The value it writes is `out_channels = in_channels`, which is exactly what the upstream fix computes — so a checkpoint carrying it loads identically before and after the fix ships. There is nothing to unwind and no version-dependent behaviour to track.

If you export by hand rather than through this harness, set the field yourself in the exported config and you get the same result.

### The serving config, and a silent failure to avoid

Both targets write this VisualGenArgs YAML:

```yaml
attention_config:
  backend: VANILLA
parallel_config:
  cfg_size: 1
  ulysses_size: 1
cuda_graph_config:
  enable: false
```

**There is no `quant_config`, and that is deliberate.** Supply one and VisualGen takes its YAML path, where `dynamic` defaults to **true**, and re-quantizes at load time — so you would be measuring a runtime-quantized model while believing you were measuring your own checkpoint. Leave it out and VisualGen reads `config_groups` from the checkpoint, which says `dynamic: false`.

That failure is silent: the model loads, generates perfectly good images, and the speedup simply never arrives. `make bench` warns when a static arm lands within 5% of BF16 for exactly this reason.

A single image by hand:

```bash
python3 <trtllm>/examples/visual_gen/models/flux1.py \
  --model $FLUX_QUANT_WORKSPACE/exports/flux-dev/hf \
  --visual_gen_args <your-config>.yaml \
  --output_path out.png
```

### Measure it on your own model

The numbers worth acting on are the ones you produce on your own weights and your own prompts. `make bench` reports per-model medians with a warm-up generation discarded — the first generation pays for CUDA context setup, kernel autotuning and allocator growth, and including it is the easiest way to report a speedup that is really a measurement error. `make serve-quality` scores the served images rather than the simulated ones, so both targets describe the artefact you would actually deploy.

Two things to know when reading the output:

**`ms/step` does not compare across models with different step counts.** Text encoding and VAE decode happen once per image, not once per step, so a 4-step model looks worse per step than a 50-step model running identical weights. Compare within a model.

**Distributional metrics are not per-image guarantees.** CMMD asks whether two *sets* of images are drawn from a similar distribution. A set can score well while containing individual images no reviewer would accept, so look at the contact sheets from `make figures` as well as the scores. Public metrics establish that the approach is sound; only your own acceptance criteria on your own model can establish that the result is good enough.

## Model arms

Both are supported and they are architecturally identical, differing only in training.

```bash
python3 quantize.py --model flux-dev      # default
python3 quantize.py --model flux-schnell
```

**Start with `flux-dev`.** It has an entry in Model Optimizer's layer-exclusion filter map and a shipped VisualGen fp4 config, so if the public path works anywhere it works there. It also runs at 50 steps, which is a common production setting.

**Then `flux-schnell`.** It is Apache-2.0, which makes it the arm to use for anything shown publicly or handed over. The catch is that `flux-schnell` has no entry in the filter map and falls back to a generic default written for a different architecture.

**That fallback costs exactly one layer.** Running Model Optimizer's own `filter_func_flux_dev` over both exports:

| Export | Result |
|---|---|
| schnell under its own fallback filter | **`proj_out` is quantized** when it should not be — 495 of 654 layers |
| the same weights under `filter_func_flux_dev` | clean, 654 of 654 checked, zero defects — 494 layers |

Since the two models are architecturally identical, forcing flux-dev's filter onto schnell's weights is the workaround — `configs/flux-schnell-devfilter.json` does exactly that. **Run `--stage schema` and read the verdict before trusting any schnell result**; it prints `VERIFIED` with a module count, or `DEFECT` naming the layers.

## Comparing two recipes on the same weights

To measure what part of a recipe is worth, you quantize the *same weights* under a *different* configuration and compare. Three things make that possible without one run destroying the other.

**`export_dir`** separates the output directory from `modelopt_model`. The model name selects Model Optimizer's layer-exclusion filter, so forcing a different filter would otherwise overwrite the first export. `configs/flux-schnell-devfilter.json` uses this to quantize schnell's weights under flux-dev's filter:

```json
"modelopt_model": "flux-dev",              // selects the filter
"baseline_dir":   "FLUX.1-schnell",        // the weights
"export_dir":     "flux-schnell-devfilter" // keeps both exports
```

**`verify` writes to `images/verify/<export_dir>/`** and generates the BF16 arm alongside, with its own `metadata.json`. Each directory is a self-contained comparison rather than depending on whatever `images/dynamic` happens to hold.

**`quality --images <dir>`** scores any such directory, and writes both `quality.json` and a `quality-<dir>.json` that the next run will not overwrite.

```bash
python3 quantize.py --config configs/flux-schnell-devfilter.json \
  --from export --through quality --prompts 16 --force
python3 quantize.py --stage quality --force \
  --images $FLUX_QUANT_WORKSPACE/images/verify/flux-schnell-devfilter
```

`--no-exclusions` does the same for the dynamic arm, quantizing every linear so the exclusions can be priced.

## Calibration is not deterministic

Model Optimizer does not fix the random seed during calibration, and the upstream README says so plainly: *"every time you run the calibration pipeline, you could get different quantizer amax values."*

Two consequences worth planning around. Two exports of the same configuration will not be byte-identical, so **do not compare checkpoints by hash**. And a quality figure carries run-to-run variance that a single export cannot show you — if a number is going to be quoted, export twice and report both.

## Workspace

Checkpoints are large, and **two different sizes get quoted for the same model because both are true.** Each FLUX.1 baseline ships the weights twice — the Diffusers sharded layout under `transformer/`, and the original single-file format at the repo root. Only one is ever loaded:

| | |
|---|---|
| On disk, per baseline | **~55 GB** |
| Actually read by a run | **~34 GB** — `transformer/` at 23.80 GB plus text encoders and VAE |

Size your volume against the first number. A full run across both arms with exports wants **250 GB**. Home directories on shared clusters are far too small.

Resolution order is `--workspace`, then `FLUX_QUANT_WORKSPACE`, then auto-detection of persistent scratch, then node-local RAID.

```bash
export FLUX_QUANT_WORKSPACE=/mnt/shared/<team>/<user>/flux-quant
```

### Sharing checkpoints across a team

Outputs are per-person, because the stages write to fixed paths and two people running at once would overwrite each other. Checkpoints are not: each baseline is ~55 GB on disk and there is no reason for a team to keep one copy each.

Create a `models` directory beside the per-person workspaces and the harness finds it on its own:

```text
/mnt/shared/<team>/
    flux-1-quantization/      the harness, one shared copy
    models/                   shared checkpoints, downloaded once
    <user>/flux-quant/        per-person outputs
```

```bash
mkdir -p /mnt/shared/<team>/models
```

That single `mkdir` is the whole opt-in. Resolution for checkpoints is `--models-dir`, then `FLUX_QUANT_MODELS`, then a `models` directory on the same scratch volume, then `<workspace>/models`. Nothing is invented — if the shared directory does not exist, each workspace keeps its own.

If your site has read-only caches that already hold these checkpoints, point at them and the download stage will copy locally instead of pulling from Hugging Face:

```bash
export FLUX_QUANT_SHARED_CACHES=/path/to/cache-a:/path/to/cache-b
```

Set `umask 002` when working on a shared volume, otherwise files land at `644` and teammates can read but not modify them. The setgid bit on the directory decides which group owns new files; the umask decides whether that group can write.

> **Node-local RAID is usually wiped when the allocation ends.** The harness warns when the workspace looks node-local, but the warning cannot save the files. If you are on RAID, copy `exports/` somewhere persistent before releasing the node, or the run produced nothing you can keep.

## Running on a Slurm cluster

Nothing here assumes Slurm, but a full run wants an uninterrupted GPU for a few hours, so an interactive allocation is the usual way to get one:

```bash
salloc --gres=gpu:1 --time=04:00:00
```

Then attach to the container and run the stages by hand. Do **not** wrap the Python entry point in `srun` inside an interactive allocation: the inherited Slurm environment makes MPI attempt a PMI bootstrap it was not built against, and the process aborts before any work starts. Attach to the container and run `python3` directly.

Blackwell B200 is `sm_100` and GB300 is `sm_103`. Behaviour differs between them, so a result on one is not evidence for the other. The manifest records which was used.

## Containers

| Stage | Container |
|---|---|
| `export` | `nvcr.io/nvidia/pytorch:26.07-py3` plus the Model Optimizer overlay |
| `dynamic`, `verify` | `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc22` |
| everything else | anywhere with the harness dependencies |

See [INSTALL.md](INSTALL.md) for the overlay steps.

## Output

Everything lands under the workspace:

```text
models/                          downloaded checkpoints, revisions pinned
exports/<export_dir>/torch/      backbone.pt — restore in PyTorch with mto.restore
exports/<export_dir>/hf/         Hugging Face export for SGLang, vLLM, TRT-LLM
images/dynamic/                  BF16 and dynamic NVFP4, paired, with metadata.json
images/verify/<export_dir>/      BF16 and static NVFP4, paired, with metadata.json
results/run_manifest.json        GPU, driver, versions, revisions, every command
results/environment.json         preflight output
results/schema_diff.json         layer-by-layer precision, filter agreement
results/tensor_inventory_*.csv   every tensor with shape and dtype
results/quality.json             most recent scoring run
results/quality-<dir>.json       one per scored directory, never overwritten
```

Every directory that can hold more than one run's output is named after what produced it. Earlier versions shared `images/verify` and a single `quality.json` across models, so a second run silently destroyed the first — and two experiments became indistinguishable on disk.

`results/run_manifest.json` records the GPU, driver, library versions, model revisions and every command as executed. Validate it before quoting anything:

```bash
python3 tools/check_run_manifest.py
```

## Metrics

Chosen to match how generated images are judged in practice rather than what is convenient to compute:

- **CMMD** is the primary distributional metric: CLIP-embedding Maximum Mean Discrepancy from [Rethinking FID](https://arxiv.org/abs/2401.09603). It is sample-efficient, so a few dozen pairs give a usable reading where FID would need tens of thousands.
- **PSNR** is a signal rather than a gate. Values well below 30 dB routinely pass human review on generative output, because two images can be far apart in pixels without either being worse.
- **CLIP score** covers text alignment, and matters disproportionately: a well-formed image with the wrong content fails, and no aggregate perceptual score catches that.
- **LPIPS** and **SSIM** are internal diagnostics only. Useful for spotting a regression, but not how generative output is usually accepted or rejected.

Results are broken out by prompt category, because the failure modes that decide sign-off — text rendering, counting, spatial relations, anatomy — disappear in an average.

**Always run a same-precision control.** Score BF16 against BF16 at different seeds and the same sample size, and you learn how far the metric moves from seed choice alone. Without that reference the number is uninterpretable: at small sample sizes the control can score *worse* than the measurement, which is the clearest possible sign you are reading noise.

No metric replaces human review, which is the actual gate in any serious evaluation. The purpose of scoring is to produce something worth putting in front of one.

## What the output actually looks like

Five pairs from the served checkpoint, BF16 left against NVFP4 right, same prompt and seed. Each carries its own PSNR and CLIP delta. All are **FLUX.1-schnell**, which is Apache-2.0; no FLUX.1-dev output appears in this repository, because that model's licence is non-commercial. Rebuild any of this on your own weights with `make figures`.

Text rendering is the first thing to break under aggressive quantization, so it is the first thing to check.

![Text rendering](docs/images/01-text-rendering.png)

![Counting](docs/images/02-counting.png)

![Photography](docs/images/03-photography.png)

Those three are the ordinary case, and the ordinary case is unremarkable. The next two are why this section exists.

![A limb the metric did not object to](docs/images/04-defect-extra-limb.png)

Two bows and a third forearm. The pair scores PSNR 12.75 dB and CLIP −0.78 — poor, but nothing in those numbers says *anatomy*, and a threshold set to catch this would reject a great many acceptable images.

![A finger the metric did not miss](docs/images/05-defect-finger-count.png)

Four fingers where BF16 renders five. This pair scores **PSNR 17.82 dB and CLIP +0.18** — unremarkable and slightly *better* than baseline on text alignment — inside an arm whose CMMD is 0.030, comfortably below a 0.042 seed-variation control. Every number is fine. The hand is not.

That last one is from the arm where the layer-exclusion filter is wrong, which is the point: a single unprotected layer produced a defect no aggregate score reported. **Look at the images.** The scores tell you where to look; they do not tell you what you will find.

## Paired comparison

Both arms are given the **same injected initial latents**, not merely the same seed. Different runtimes consume the random stream differently, so two stacks handed seed 42 can start from different noise. Latents are generated once on CPU, hashed, and the hash is written into the image metadata so a reviewer can verify the pairing rather than assume it. The `dynamic` stage fails outright if the hashes do not match across arms.

## Known traps

- `--format fp4`, not `nvfp4`. There is no `nvfp4` value; `fp4` selects the NVFP4 preset. With `--quantize-mha` that means NVFP4 linear layers **and FP8 attention**, which is *not* what the dynamic runtime path does — see Scope.
- `--quantized-torch-ckpt-save-path` is treated as a **directory**, although the upstream example still passes a name ending in `.pt`. Passing a filename silently creates a directory of that name.
- Without `--quantize-mha`, attention stays at higher precision.
- FLUX.1 checkpoints are gated on Hugging Face. Accept the licence in a browser and hold a token before downloading. Preflight checks this.

## Scope

The pipeline is mixed precision, never 4-bit throughout — and **the two arms differ on attention**, which matters more than it sounds.

| | Linear layers | Attention | Everything else |
|---|---|---|---|
| **Dynamic** (VisualGen, torchao) | NVFP4 weights and activations, W4A4 | **BF16**, W16A16 | BF16 |
| **Static export** (`--quantize-mha`) | NVFP4 weights and activations | FP8 | BF16 |

Text encoders, the VAE, normalization, and the embedding and projection layers stay at BF16 in both. The accumulator stays at BF16 so addition does not drop low bits.

Attention is left alone in the dynamic path deliberately, to protect quality. Keeping attention at FP8 while the rest goes to NVFP4 is known from LLM work to cost little or nothing, so it is the next thing to try rather than the current state — which is why our static export, which does quantize attention, is not a like-for-like comparison against a dynamic run.

This is also the answer to "why didn't halving the precision halve the runtime". It did not halve the work: only the linear layers moved, so the speedup is bounded by the fraction of end-to-end time they account for. Modest end-to-end speedups from FP8 on this pipeline follow from the same arithmetic.

## Licensing

**FLUX.1-schnell is Apache-2.0** and can be used for personal, scientific and commercial purposes. **FLUX.1-dev is under the FLUX.1 [dev] Non-Commercial License**, as is the published FLUX.1-dev-NVFP4 checkpoint — it can be downloaded after accepting the terms, but not used as the basis of a commercial deployment.

**Use the `flux-schnell` arm for anything you intend to publish, share or deploy commercially.** It is architecturally the same model and reaches the same result. `flux-dev` is included because it has an entry in Model Optimizer's layer-exclusion filter map and runs at 50 steps, which makes it the better subject for a like-for-like comparison — not because it is something you would ship.

The recipe itself carries no such constraint. It is public Model Optimizer configuration, and the licence attaches to the weights and their outputs rather than to the method.
