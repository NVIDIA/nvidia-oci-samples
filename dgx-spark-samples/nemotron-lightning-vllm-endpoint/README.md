<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
-->

# Nemotron 3.5 Lightning vLLM Endpoint on DGX Spark

This sample shows how developers can serve [NVIDIA Nemotron 3.5 Lightning 30B-A3B (NVFP4)](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4) on a single NVIDIA DGX Spark as a private, OpenAI-compatible endpoint, then drive it from the standard OpenAI Python SDK.

It is intentionally small and external-safe:

- No API keys, credentials, customer data, or internal NVIDIA content.
- No OCI account required — this sample runs entirely on local DGX Spark hardware.
- All model and inference components are publicly available open-source or open-weights releases.
- Agent tool implementations are deterministic local functions with synthetic data. Nothing in the demo reaches the network at runtime.

The point of the sample is a single line:

```python
client = OpenAI(
    base_url="http://localhost:8000/v1",   # <- the only line that changes
    api_key="not-needed",                  #    no key: it is your hardware
)
```

Any tool built on the OpenAI API — LangChain, LlamaIndex, Continue, or your own services — points at a model running on hardware you own by changing configuration, not code.

## What the Sample Shows

1. Brings up an open-weights 30B model on one DGX Spark with a single command.
2. Confirms the endpoint is healthy and reports the served model and context length.
3. Drives it from the OpenAI Python SDK by changing `base_url`.
4. Streams a response while reporting live time-to-first-token and decode rate.
5. Runs a multi-step tool-calling agent that chains two unrelated tools without being told to.
6. Reports remaining memory headroom on the machine.
7. Points at the fine-tuning and quantization workflows that use the same box.

## Why This Goes Beyond A Chat Demo

- **Structured tool calling, not text scraping.** The model is served with `--enable-auto-tool-choice --tool-call-parser qwen3_coder`, so tool calls arrive as structured `tool_calls`.
- **Structured reasoning traces.** `--reasoning-parser nemotron_v3` returns the reasoning trace as a field rather than inline prose.
- **Multi-step planning.** Given a flight-search tool and a currency-conversion tool and a task needing both, the model works out that it must search first and convert second.
- **Honest performance characterisation.** The notebook reports measured TTFT and decode rate rather than asserting them, and states plainly where a single DGX Spark is the wrong tool.

## Everything Here Is Open

| Component | License |
| --- | --- |
| [vLLM](https://github.com/vllm-project/vllm) | Apache 2.0 |
| [Nemotron 3.5 Lightning 30B-A3B NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4) | OpenMDW-1.1 |
| [nvidia/When2Call](https://huggingface.co/datasets/nvidia/When2Call) | CC-BY-4.0 |
| This sample | Apache 2.0 |

NVIDIA released the **pre-training and post-training data** alongside the Nemotron weights, not only the weights. For teams evaluating whether to build on a model, that is a materially different position from a weights-only release.

No NVIDIA licence, NGC subscription, or support contract is required to run this sample.

## Requirements

- NVIDIA DGX Spark (GB10, `sm_121`, 128 GB unified memory) running DGX OS.
- Docker available to your user.
- Python 3.10 or newer.
- Approximately 45 GB of free disk: ~20 GB of model weights plus the ~11 GB compressed
  (larger unpacked) vLLM container image, with headroom.
- A Hugging Face token (`HF_TOKEN`) only if you hit download rate limits.

**No OCI account and no cloud GPU shape are required.**

The first run does need internet egress, to `docker pull` the vLLM image and fetch the weights
from Hugging Face — roughly 31 GB in total. After both are cached on the device, serving and the
whole local half of the notebook run with no egress at all. The hosted-model comparison in
section 4 is the exception: it calls `integrate.api.nvidia.com` by definition, and it is the one
part you can skip on an isolated machine.

## Connecting to Your DGX Spark

You can work directly on the device with a keyboard and monitor, but most people drive it from a laptop. There are two ways to do that, and the first is considerably easier.

### Option A: NVIDIA Sync (recommended)

[NVIDIA Sync](https://docs.nvidia.com/dgx/dgx-spark/nvidia-sync.html) is a free utility for macOS, Windows, and Ubuntu that manages SSH connections, port forwarding, and tunnels for you. It is the fastest way to get from a laptop to a working JupyterLab on the device.

1. **Install it** from [build.nvidia.com/spark/connect-to-your-spark/sync](https://build.nvidia.com/spark/connect-to-your-spark/sync).
2. **Add your device.** On the same network, Sync discovers DGX Spark systems automatically over mDNS. Otherwise add it by hostname or IP, using the account credentials you created during first boot. Sync configures SSH key-based authentication for you, so subsequent connections need no password. See [Direct Connections](https://docs.nvidia.com/sync/latest/direct-connections.html).
3. **Connect**, then launch **DGX Dashboard**, **Terminal**, or **VS Code** from the app list. Each opens against the device with tunnels already in place.

Once Sync has added the device it writes a **Host alias into `~/.ssh/config`**, so you can also
reach it from any terminal without opening the Sync app:

```bash
ssh -t <sync-host-alias> "cd ~; bash -l"
```

Two flags worth understanding, because the bare `ssh <alias>` form causes confusing failures later:

- **`-t`** forces a TTY. Without it you get a non-interactive session, and anything that expects a
  terminal — progress bars, `docker` output, `Ctrl-C` on a foreground server — misbehaves.
- **`bash -l`** starts a *login* shell, which sources your profile. Without it `PATH` may be
  missing entries added at login, so tools that work when you sit at the machine are "not found"
  over SSH.

Run `grep -i host ~/.ssh/config` to find the alias Sync created if you do not know it.

**If your laptop is not on the same network as the device**, Sync supports [Tailscale connections](https://docs.nvidia.com/sync/latest/tailscale.html) — useful for a Spark sitting on an office network while you work remotely.

Full walkthrough: [NVIDIA Sync Getting Started](https://docs.nvidia.com/sync/latest/getting-started.html).

### Running this notebook through the DGX Dashboard

The [DGX Dashboard](https://docs.nvidia.com/dgx/dgx-spark/dgx-dashboard.html) has an integrated JupyterLab, and NVIDIA Sync tunnels it automatically. Connect in Sync, click **DGX Dashboard** (it opens at `http://localhost:11000`), set the working directory to this sample, and start JupyterLab. Then open `demo.ipynb`.

> **Important if you take this route.** The Dashboard's JupyterLab creates a **fresh virtual environment per working directory**, which is a *different* environment from the `.venv` that `setup.sh` builds — so the sample's dependencies will be missing. Two ways out, and the first is cleaner:
>
> - **Select the "DGX Spark demo" kernel** in JupyterLab. `setup.sh` registers `.venv` under that name, so the notebook runs against the right environment with nothing to install.
> - Or install into the Dashboard's environment from a cell: `%pip install -r requirements.txt`. Changing the working directory spawns another new environment, so you would repeat this each time.

### Option B: plain SSH

```bash
ssh <username>@<device-name>.local
```

**The `.local` suffix is required** when connecting by device name — DGX Spark advertises itself over mDNS as `spark-xxxx.local`, and `ssh user@spark-xxxx` will fail to resolve. Use the IP address instead if mDNS is blocked on your network.

To reach JupyterLab or the Dashboard this way you must tunnel the ports yourself:

```bash
# DGX Dashboard
ssh -N -L 11000:localhost:11000 <username>@<device-name>.local

# JupyterLab started manually by this sample
ssh -N -L 8888:localhost:8888 <username>@<device-name>.local
```

`-N` means "set up the forward, do not run a command". The terminal will sit there with no prompt
and no output — that is what success looks like. Leave it running for as long as you need the
tunnel. If Sync gave you a Host alias, use it in place of `<username>@<device-name>.local`.

The Dashboard's integrated JupyterLab assigns a **per-user port**, listed in `/opt/nvidia/dgx-dashboard-service/jupyterlab_ports.yaml` on the device — check there before tunnelling. NVIDIA Sync handles all of this for you, which is why Option A is recommended.

### Opening the notebook over a tunnel

Whichever way you connect, this is the most reliable way to run `demo.ipynb` — the activated
environment *is* the kernel, so there is no kernel to pick and no way to pick the wrong one.

```bash
# On the DGX Spark, in a new terminal. Leave serve.sh running in its own.
cd <clone>/dgx-spark-samples/nemotron-lightning-vllm-endpoint
source .venv/bin/activate
jupyter lab --no-browser --ip 127.0.0.1 --port 8888
```

```bash
# On your laptop, in another terminal. Leave it running; it prints nothing.
ssh -N -L 8888:localhost:8888 <username>@<device-name>.local
```

Then open the tokenised URL Jupyter printed.

Two details that matter:

- **`source .venv/bin/activate` is required.** `jupyter` is installed in `.venv`, not on the
  system PATH, so without it the command is not found.
- **`--ip 127.0.0.1` binds Jupyter to loopback**, so it is reachable only through your tunnel.
  Without it Jupyter listens on every interface, and anyone on the same network who can read the
  token has your notebook.

### Nothing in this sample needs editing

Clone it, run `setup.sh`, run `serve.sh`, open `demo.ipynb`. There is no configuration step and
no file to change first.

`demo.ipynb` uses `base_url="http://localhost:8000/v1"`, and that is correct **because the
notebook runs on the DGX Spark itself** — the kernel and vLLM are on the same machine, so port
8000 is genuinely local. You may be *viewing* JupyterLab through a tunnel from a laptop, but the
Python is executing on the Spark.

**The one case where you do change it:** if you run the notebook (or any other client) on your
laptop rather than on the Spark, `localhost` points at the wrong machine. Open an SSH tunnel and
keep pointing at `localhost`:

```bash
ssh -N -L 8000:localhost:8000 <username>@<device-name>.local
```

```python
BASE_URL = "http://localhost:8000/v1"     # tunnelled to the Spark
```

The tunnel is the supported route, and `serve.sh` binds the published port to `127.0.0.1` for
that reason: **vLLM serves no authentication of any kind.** Anything that can reach the port can
use the model, read whatever is sent through it, and occupy the GPU. Over a tunnel the traffic is
carried inside SSH; pointed at `http://spark-xxxx.local:8000` it crosses your network in
cleartext, open to anyone who can route to the device.

If you have a specific reason to expose it on a trusted segment, opt in explicitly:

```bash
BIND_ADDR=0.0.0.0 ./serve.sh
```

and put an authenticated TLS reverse proxy in front of it. The `.local` suffix is required if you
address the device by name — DGX Spark advertises over mDNS, and the bare hostname will not
resolve; use the IP address if mDNS is blocked on your network.

Separately, and not needed to run any of this: the `base_url` line is also how you point an
application *you already have* at this endpoint. That is covered in section 3 of the notebook, and
it is about your own code on another day — not a step in getting this sample working.

## Quickstart

Verify local prerequisites:

```bash
nvidia-smi
docker info >/dev/null && echo "docker OK"
python3 --version
```

If `docker info` fails, add yourself to the docker group and start a new login shell:

```bash
sudo gpasswd -a $(whoami) docker
```

Clone the sample:

```bash
git clone -b feature/nemotron-lightning-vllm-endpoint \
  https://github.com/damienjose/nvidia-oci-samples.git
```

> **Why a branch, and why that fork?** This sample is under review for
> [NVIDIA/nvidia-oci-samples](https://github.com/NVIDIA/nvidia-oci-samples). Until it merges, the
> code lives on the feature branch above. Once merged, the clone command becomes simply
> `git clone https://github.com/NVIDIA/nvidia-oci-samples.git` with no branch flag — check the
> upstream repository first, and use the fork only if the `dgx-spark-samples/` directory isn't
> there yet.

Run the sample:

```bash
cd nvidia-oci-samples/dgx-spark-samples/nemotron-lightning-vllm-endpoint
./setup.sh                    # environment, container, weights
source .venv/bin/activate     # required — see below
./serve.sh                    # start vLLM (leave running; ~4 min cold start)
```

**`setup.sh` creates a virtual environment at `.venv` and installs into it.** DGX OS ships a
PEP 668 "externally managed" Python, so installing into the system interpreter is blocked by
design. Activate `.venv` before running the notebook, `run_benchmark.py`, or `make_chart.py` —
if any of them reports a missing module, that is the reason. `serve.sh` is the exception: it
only needs Docker.

`setup.sh` is **safe to re-run**. It checks the local Hugging Face cache first with no network call, so weights you already have are never re-downloaded — a second run takes seconds. A first run measured **3 min 39 s end to end** on a wired connection, of which 3 min 18 s was the ~20 GB download; it is bandwidth-bound, so expect longer on a slower link. The download resumes if interrupted.

In a second terminal:

```bash
jupyter lab demo.ipynb
```

Or open `demo.ipynb` from the DGX Dashboard's integrated JupyterLab — see [Connecting to Your DGX Spark](#connecting-to-your-dgx-spark), and note the virtual-environment caveat there.

Verify the endpoint from the shell at any time:

```bash
curl -s http://localhost:8000/v1/models | python3 -m json.tool
```

## Files

| File | Purpose |
| --- | --- |
| `setup.sh` | Machine check, client dependencies, model pre-download |
| `serve.sh` | Launch vLLM with the flags this model requires |
| `demo.ipynb` | Walkthrough: environment, endpoint, SDK switch, streaming, agent loop, headroom |
| `demo_tools.py` | Tool implementations and JSON schemas for the agent section |
| `when2call.py` | Benchmark scoring — dataset loading, metrics, Wilson intervals |
| `preflight.py` | Check every endpoint answers, serves the right model, and returns structured tool calls — before you sweep |
| `run_benchmark.py` | Run When2Call against one or more endpoints, with 429 backoff |
| `make_chart.py` | Chart `results/summary.json` |
| `endpoints.json` | Endpoint config — the local Spark plus four models on the public NVIDIA API catalog. Keys come from the environment, never committed |
| `results/` | Benchmark output — `summary.json` and generated charts |
| `requirements.txt` | Client-side Python dependencies, installed into `.venv` |

## Serving Configuration

Four flags. This is the configuration the published When2Call results on this hardware came from —
120/120 examples scored, zero errors — and it is what `serve.sh` runs by default.

```bash
vllm serve nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 \
  --max-model-len 65536 \
  --reasoning-parser nemotron_v3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --host 0.0.0.0 --port 8000
```

`--host 0.0.0.0` is the bind *inside the container*, which is how Docker reaches the process;
`serve.sh` limits the exposure on the host side with `-p 127.0.0.1:8000:8000`. If you run this
command directly on the device instead of through `serve.sh`, use `--host 127.0.0.1` — otherwise
you are publishing an unauthenticated endpoint on every interface.

| Flag | Why |
| --- | --- |
| `--max-model-len 65536` | 64k context. The model supports up to 1M; 64k is what was measured |
| `--reasoning-parser nemotron_v3` | Returns the reasoning trace as a structured field |
| `--enable-auto-tool-choice` | Enables model-initiated tool calls |
| `--tool-call-parser qwen3_coder` | Returns tool calls as structured `tool_calls` |

### Three flags deliberately not passed

Worth stating, because they appear in other write-ups and look like omissions here.

- **`--kv-cache-dtype fp8`** — this ModelOpt checkpoint pins fp8 KV in its own quantisation
  config, so vLLM selects `fp8_e4m3` whether or not you pass it. Verified on this hardware.
- **`--moe-backend marlin`** — GB10 has no native FP4 compute, so vLLM already falls back to
  Marlin and logs that it did. Pinning it by hand adds nothing here and would override a better
  default on hardware that does have FP4.
- **`--trust-remote-code`** — the checkpoint declares `nemotron_h` with no `auto_map`. There is no
  remote code to load, so this only widens what would execute if the repo were ever compromised.

### Tuned profile

`TUNED=1 ./serve.sh` adds NVIDIA's fuller DGX Spark configuration from the
[vLLM day-0 announcement](https://vllm.ai/blog/2026-08-10-nemotron-3-5-lightning-vllm) — explicit
backends, Mamba state-space cache tuning, prefix caching, a larger batched-token budget. It is a
sound configuration published by the people who built the model.

It is not the profile these numbers came from, and **two of its flags change what you are
measuring**, so do not mix profiles within a comparison:

- **`--enable-prefix-caching`** is a confound for latency work here. All 120 When2Call examples
  share a system prompt and tool schemas, so prefix caching turns most prefills into cache hits.
  Time-to-first-token improves for a reason that has nothing to do with the model, and the
  local-versus-hosted latency comparison stops being like-for-like.
- **`--mamba-ssm-cache-dtype float16` with stochastic rounding** changes numerics in the
  state-space path. Accuracy results are only comparable against a run configured the same way.

Benchmark on the default profile. Use `TUNED=1` to explore serving performance, and say which
profile produced any number you publish.

## Credentials

**`.venv` and `.env` are not the same thing, and only one of them exists here.**
`setup.sh` creates `.venv/`, a Python virtual environment you activate with
`source .venv/bin/activate`. There is no `.env` file — `.env` and `*.key` appear in `.gitignore`
purely as a guard against someone creating one. Keys stay in the environment.

Three different things get confused here, so to be explicit — **the core demo needs none of them.**

| | Needed for | When |
| --- | --- | --- |
| **nothing** | Serving the model, the notebook, the local benchmark | Always. The weights and the dataset are public. |
| `HF_TOKEN` | Hugging Face downloads | **Optional.** Only if you hit anonymous rate limits, which usually happens on shared or heavily-NAT'd networks. |
| `NVIDIA_API_KEY` | The four *hosted* models in the five-endpoint benchmark sweep | Only for `./run_benchmark.py` without `--only spark`. A free NGC personal API key from [build.nvidia.com](https://build.nvidia.com/) — not a Hugging Face token. |

**Anyone can get this key.** Create an NGC account at [ngc.nvidia.com](https://ngc.nvidia.com),
generate a Personal API Key with the **NVIDIA Public API Endpoints** service enabled, and it works
against `https://integrate.api.nvidia.com/v1` — the base URL printed on every model card at
build.nvidia.com. The free tier is rate limited, which is why `endpoints.json` uses low
concurrency; higher throughput comes with NVIDIA AI Enterprise.

> **A note if you work at NVIDIA.** Internal inference services exist with larger catalogues and
> their own keys. This sample deliberately does **not** use them: a benchmark nobody outside
> NVIDIA can re-run is not reproducible, and reproducibility is the point. If you want to run the
> sweep against an internal service, copy `endpoints.json`, change `base_url` and the model ids,
> and keep the copy local — `endpoints-*.json` is gitignored for that purpose. Note that internal
> model ids usually carry a provider prefix the public catalog does not.

The model ids in `endpoints.json` are the catalog's own, taken from each model card. They differ
from the Hub's, which carry a provider prefix — `openai/gpt-oss-120b` here versus
`nvcf/openai/gpt-oss-120b` there. Ids are not portable between the two services.

```bash
curl -s https://integrate.api.nvidia.com/v1/models \
  -H "Authorization: Bearer $NVIDIA_API_KEY" | python3 -c "
import json,sys
ids = [m['id'] for m in json.load(sys.stdin)['data']]
print(len(ids), 'models visible to this key')
for want in ['nvidia/nemotron-3.5-lightning-30b-a3b', 'openai/gpt-oss-120b',
             'google/gemma-4-31b-it', 'moonshotai/kimi-k3']:
    print(('  ok   ' if want in ids else '  MISSING  ') + want)"
```

Both are read from the environment and never written to disk by this sample. `endpoints.json`
references `NVIDIA_API_KEY` by name, not by value, so nothing secret is committed.

### Setting a key without putting it on screen

`export NVIDIA_API_KEY=nvapi-...` types the key in the clear and saves it to your shell history.
Fine alone at a desk; not fine if you are recording, pairing, or presenting. Two better ways:

```bash
# Prompted, never echoed, never in history. bash and zsh disagree on read -p,
# so this uses getpass, which behaves the same in both.
NVIDIA_API_KEY=$(python3 -c 'import getpass; print(getpass.getpass("NVIDIA_API_KEY: "))')
export NVIDIA_API_KEY

# bash only -- in zsh, read -p means "read from a coprocess" and this fails
# with "read: -p: no coprocess"
read -rs -p "NVIDIA_API_KEY: " NVIDIA_API_KEY && export NVIDIA_API_KEY && echo

# Or keep it in a file outside the repo, readable only by you
install -m 600 /dev/null ~/.nvidia_api_key   # create it 0600 from the start,
                                             # then paste the key in with an editor
export NVIDIA_API_KEY="$(cat ~/.nvidia_api_key)"
```

Confirm it took, without printing it:

```bash
[ -n "$NVIDIA_API_KEY" ] && echo "set, ${#NVIDIA_API_KEY} chars" || echo "not set"
```

**Never set a key in a notebook cell.** `os.environ["NVIDIA_API_KEY"] = "nvapi-..."` is saved into
the `.ipynb`, survives "clear outputs" because it is source rather than output, and is then one
`git add` away from being public. The environment the kernel was launched with is the right place.

### If you are presenting this

**The live benchmark cell needs no key at all.** Notebook section 7c scores a small slice against
your own Spark, using the same `client` as every other cell — `api_key="not-needed"`. Only
`run_benchmark.py` reaching the four *hosted* models needs a credential, and that is a ten-minute
sweep you would run beforehand, not on stage.

So the sequence that keeps a key off screen entirely is: set it in a terminal before the session,
run the full sweep, and let the notebook read the resulting `results/summary.json`. On the day,
nothing you run in front of anyone touches a credential.

Three places a key leaks that people forget: **terminal scrollback** (if you exported it earlier in
the window you are now sharing, scrolling up reveals it), **`env` or `printenv`** run on camera,
and **shell history** recalled with an up-arrow in front of an audience.

**Verify the dataset needs no token** — takes a few seconds:

```bash
python3 -c "
from datasets import load_dataset
d = load_dataset('nvidia/When2Call', 'test', split='mcq')
print('OK —', len(d), 'rows, no token required')"
```

A `401`/`403` or a *"gated dataset"* message means you need to accept the terms on the
[dataset page](https://huggingface.co/datasets/nvidia/When2Call) while logged in, then export
`HF_TOKEN`. Otherwise no credential is involved.

A `ValueError` naming the config or the split means the layout moved again. It has moved once
already — what used to be config `mcq` with a `test` split is now config `test` with an `mcq`
split, which is why the line above reads back-to-front. `when2call.py` resolves both from the Hub
at load time rather than hard-coding them, and records what it used in `results/summary.json`, so
the benchmark keeps working when this one-liner stops.

## Environment Variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `MODEL` | `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4` | Model to serve |
| `MODEL_REVISION` | unset (`main`) | Pin a specific Hugging Face revision. Set this if you need byte-identical weights across machines |
| `PORT` | `8000` | Host port for the endpoint |
| `MAX_MODEL_LEN` | `65536` | Context length |
| `VLLM_IMAGE` | `vllm/vllm-openai:v0.27.1` | Container image. Multi-arch, includes `linux/arm64`. Must be a release with Nemotron 3.5 Lightning support — see Known Issues |
| `VLLM_CACHE_DIR` | `~/.cache/vllm` | Persisted torch.compile and FlashInfer autotune caches |
| `SKIP_PRECHECK` | unset | Set to `1` to skip the architecture-support probe in `serve.sh` |
| `TUNED` | unset | Set to `1` to add NVIDIA's DGX Spark tuning flags. Changes what latency and accuracy runs measure — see Serving Configuration |
| `CONTAINER_NAME` | `vllm-nemotron` | Docker container name |
| `HF_HOME` | `~/.cache/huggingface` | Hugging Face cache location |
| `HF_TOKEN` | unset | Only needed if you hit download rate limits |
| `VENV_DIR` | `.venv` in the sample directory | Where `setup.sh` creates the virtual environment |

## Benchmarking with When2Call

Serving a model is easy to demo and hard to trust. This sample includes a reproducible
evaluation so the claim is checkable on your own hardware.

[`nvidia/When2Call`](https://huggingface.co/datasets/nvidia/When2Call) (CC-BY-4.0) asks a
harder question than most function-calling benchmarks: not "was the call correct" but
**"should it have called anything at all"**. Three labels — call the tool, ask for the
missing argument, or decline because no tool fits.

Scoring is deterministic. We observe what the server did; there is no LLM judge, so the
numbers reproduce.

### Results ship with the repo; re-running them is optional

`results/summary.json` and the charts are committed, so `demo.ipynb` renders the comparison with
no key, no network and no waiting. That is deliberate: the sweep is 600 requests across five
endpoints and takes a few hours, most of it spent waiting on shared free-tier gateways rather than
on model compute. Fine unattended; wrong in front of an audience.

Nothing about it is precomputed magic, though — the committed file is the output of
`run_benchmark.py` in this directory, and you can regenerate it.

### Run it yourself

**1. Get a free NVIDIA API key.** Create an account at [build.nvidia.com](https://build.nvidia.com/),
then generate an **NGC Personal API Key** with the *NVIDIA Public API Endpoints* service enabled.
No NVIDIA affiliation is required and the free tier is sufficient.

**2. Export it** without leaving it in your shell history:

```bash
read -rs -p "NVIDIA_API_KEY: " NVIDIA_API_KEY && export NVIDIA_API_KEY && echo
```

**3. Check the endpoints first.** One request each, a few minutes, and it catches the failures that
would otherwise waste the whole run:

```bash
./preflight.py --real 2
```

**4. Run it.**

```bash
./run_benchmark.py --config endpoints.json --n 40 --save-raw
python3 make_chart.py
```

Results are checkpointed after every endpoint, so an interrupted run keeps what it had; `--resume`
picks up from there. Budget a few hours for all five.

**Only want the local number?** `--only spark` finishes in about twelve minutes and **needs no key
at all** — it talks to the model on your own machine. That is the figure this sample is really
about, and it is the one nobody can rate-limit.

From inside the notebook, `FORCE_RERUN = True` in section 7 does the same thing. Jupyter inherits
the environment of the shell that launched it, so export the key *before* starting Jupyter or the
hosted endpoints will be skipped.

### Metrics

| Metric | What it tells you |
| --- | --- |
| Decision accuracy | Did it call exactly when it should have? |
| Actionable decision accuracy | Correct decision *and* it either called or produced real text. Stops a model scoring well by staying silent. |
| Tool-selection accuracy | When it correctly called, was it the right tool? |
| **Over-call rate** | How often it fired a tool it shouldn't have. **Lower is better** — this is the number that predicts agent misbehaviour in production. |

### Reading results honestly

Two things the harness will not do for you:

- **At n=120 the interval is roughly ±6.5 points.** Models within about 10 points of each
  other are a statistical tie, not a ranking. `when2call.overlaps()` tells you when two
  confidence intervals overlap; the notebook uses it to say so out loud.
- **A model that returns no data is reported as "no data", never as a low score.** A
  rate-limited endpoint is a missing measurement.

### Run it somewhere a dropped SSH session cannot kill it

`docker run` in the foreground proxies signals to the container, so if the SSH session running
`serve.sh` drops, the endpoint goes with it — and `--rm` means the container is gone, not stopped.
The same applies to a sweep: an hour of benchmarking dies with the connection.

```bash
./serve.sh --detach                       # container survives the shell
docker logs -f vllm-nemotron              # follow it; Ctrl-C detaches, does not stop

nohup ./run_benchmark.py --config endpoints.json --n 40 --save-raw \
      > ~/bench.log 2>&1 &                # sweep survives the shell
tail -f ~/bench.log
```

`tmux` or `screen` work equally well and let you reattach. Whichever you pick, do not run a
long job in a bare foreground SSH session — especially not the endpoint you are about to demo.

### Free-tier endpoints queue, and the wait is not generation time

Measured on `integrate.api.nvidia.com`: one model returned **17 tokens in 264 seconds**. That is
not a slow model, it is a queue on a shared free-tier endpoint. Two consequences:

- **Lowering `max_tokens` does not help.** Almost none of that time was generation.
- **Raising `concurrency` does.** Requests wait in parallel rather than in series. `endpoints.json`
  sets high concurrency and long timeouts for the endpoints that behave this way, and the retry
  backoff absorbs the 429s that come with it.

`preflight.py` projects a single request's latency onto the full sweep, so you find out that an
endpoint would take four hours before you start it, rather than forty minutes in.

### Rate limits are the main failure mode

An earlier sweep issued 960 requests at concurrency 4 and tripped the hosted gateway:
four of eight models returned HTTP 429 on *every* call, including control requests with
no tools attached, and produced nothing. `run_benchmark.py` retries 429 and 5xx with
exponential backoff and full jitter, and defaults remote endpoints to low concurrency.
If a hosted model still returns no data, lower it further:

```bash
./run_benchmark.py --concurrency 1 --n 40
```

### The local-vs-hosted control

`endpoints.json` runs Nemotron 3.5 Lightning in **two places** — on the Spark and on the
API catalog. That pair is the control for the whole comparison: if the same model
scored differently in the two locations, the cross-model numbers would mean nothing.

## Measured On One DGX Spark

Measured on `spark-c251` (GB10, `sm_121`, driver 580.159.03) with
`vllm/vllm-openai:v0.27.1` and snapshot `cc84af2f…`.

| Metric | Value |
| --- | --- |
| Weights on disk | 20.1 GB across 70 files |
| Weights resident | 17.85 GiB |
| **KV cache** | **90.08 GiB — 24,703,795 tokens** |
| Maximum concurrency at 64k context | 376.95× |
| Peak activation | 2.23 GiB |
| CUDA graph pool | 0.44 GiB |
| First-run weight download | 3 min 18 s (~100 MB/s wired) |
| `setup.sh`, everything cached | 3.7 s |
| Cold start to `Application startup complete` | ~4 min |

Where that four minutes goes, since it is the number people plan around:

| Phase | Time |
| --- | --- |
| Loading 52 safetensors shards | 100.4 s |
| Engine init — profile, KV cache, warmup | 93.0 s |
| ↳ Mamba2 SSD Triton kernel warmup | 35.2 s |
| ↳ FlashInfer fp8 GEMM autotune | 18.0 s |
| ↳ CUDA graph capture (86 graphs) | 13.0 s |
| ↳ `torch.compile` | 7.5 s |

`serve.sh` mounts `~/.cache/vllm` into the container, so the `torch.compile`
artefacts and the FlashInfer autotune results survive a restart — about 25 s
that only the first start pays.

**Three things the startup log confirms**, each of which is asserted elsewhere in this README:

- `Using fp8_e4m3 data type to store kv cache` appears with no `--kv-cache-dtype` passed. The
  checkpoint pins it, exactly as documented.
- `Using 'MARLIN' NvFp4 MoE backend` is selected automatically from eight candidates, alongside
  `Your GPU does not have native support for FP4 computation`. GB10 has no FP4 tensor cores;
  NVFP4 is buying memory here, not compute.
- `Updating mamba_ssm_cache_dtype to 'float32' for NemotronH` — vLLM overrides this itself. Worth
  knowing before you hand-set `--mamba-ssm-cache-dtype float16`, as the tuned profile does.

**The weights moved.** An earlier run of this benchmark used snapshot `b14872a5…` at 21.6 GB
across 69 files; the current `main` resolves to `cc84af2f…` at 20.1 GB across 70 files. Same repo,
same model name, different bytes. Accuracy numbers do not carry across a snapshot change, which is
why `setup.sh` prints the snapshot id and asks you to record it. Pin `MODEL_REVISION` if you need
a run to stay reproducible.

Serving figures — TTFT and decode rate — are measured live by `demo.ipynb` rather than asserted
here, because they depend on the serve flags above and on what else is resident on the machine.
Run the notebook and read your own numbers.

Expect prefill to be cheap relative to decode on GB10: at 273 GB/s the decode path is
memory-bandwidth-bound, so time-to-first-token stays close to flat as the prompt grows while
tokens per second does not.

### How long from `git clone` to a live endpoint

| | Cold, nothing cached | Everything cached |
| --- | --- | --- |
| `git clone` | ~5 s | ~5 s |
| `setup.sh` — venv and client deps | ~1 min | ~10 s |
| `setup.sh` — pull the vLLM image (~10.5 GB compressed) | 3–6 min | 0 |
| `setup.sh` — download weights (20.1 GB) | ~3.5 min | 0 |
| `serve.sh` — cold start | ~4 min | ~3.5 min |
| **Total** | **~12–15 min** | **~4 min** |

Roughly 31 GB crosses the network on a first run, so the cold figure is bandwidth-bound and can be
considerably longer on a slow link. **Start `setup.sh` before you need the endpoint**, not while
someone is watching.

## Demo Flow

For a 5 to 10 minute presentation:

1. Show the clone command and invite the audience to run it on their own DGX Spark.
2. Run the environment cell to confirm GB10, 128 GB unified memory, `sm_121`.
3. Confirm the endpoint is healthy.
4. Change `base_url` and run the same OpenAI SDK code against local hardware.
5. Stream a response with live TTFT and decode rate visible.
6. Take a prompt from the audience and run it.
7. Run the agent loop and point out that nobody told it to chain the two tools.
8. Show remaining memory headroom.
9. Close on the fine-tuning and quantization workflows that use the same machine.

## Scope And Limitations

Stated plainly, because sizing this box correctly matters more than overselling it:

- **A single GB10 is a development and prototyping endpoint, not a shared production one.** With 273 GB/s of memory bandwidth, concurrent requests contend and largely serialise. One developer or one CI job is an excellent fit; dozens of simultaneous users are not.
- **GB10 has no native FP4 compute.** vLLM logs `Your GPU does not have native support for FP4 computation` and falls back to the Marlin kernel. NVFP4 here is a *memory* optimisation — weights stored 4-bit and decompressed to compute. A large win on a bandwidth-bound part, but it is not FP4 tensor-core acceleration.
- **Do not select this hardware for throughput.** Its defensible advantages are capacity, data residency, zero marginal cost per token, and deterministic capacity with no rate limits.

## Going Faster: Speculative Decoding

Nemotron 3.5 Lightning ships three speculative decoders — multi-token prediction (MTP), DFlash,
and DSpark — and NVIDIA reports DSpark as the fastest of the three on DGX Spark. Speculative
decoding drafts several tokens ahead and has the target model verify them in parallel, cutting
sequential decode steps without changing what the model outputs.

Not enabled here, for two reasons: DSpark needs a second checkpoint, so turning it on costs
another download; and published acceptance-length figures (1.808 for MTP, 2.83 for DSpark) are
**unverified on this hardware** — treat them as something to measure, not to quote.

```bash
--speculative_config.method dspark \
--speculative_config.model nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark \
--speculative_config.num_speculative_tokens 3
```

Measure before and after on your own workload. Speculative decoding helps most where drafts are
predictable — structured output, tool-call arguments, code — and least on short, high-entropy
replies. Full configurations for MTP, DFlash, DSpark, H100, and Jetson are in the
[vLLM day-0 announcement](https://vllm.ai/blog/2026-08-10-nemotron-3-5-lightning-vllm).

## Known Issues

### DGX OS has an externally-managed Python (PEP 668)

`pip install` into the system interpreter fails with `error: externally-managed-environment`.
This is deliberate on Debian-derived systems — the OS owns those packages.

`setup.sh` handles it by creating a virtual environment at `.venv` and installing there, then
registering it with Jupyter as a kernel named **DGX Spark demo**. Two consequences:

- **Activate before running anything Python:** `source .venv/bin/activate`
- **In JupyterLab, select the "DGX Spark demo" kernel**, not the default one

If `python3 -m venv` itself fails, the module ships separately on Ubuntu:
`sudo apt install -y python3-venv`.

Resist `pip install --break-system-packages`. It works, and it can break OS-managed packages on
a machine you would rather keep working.

### The vLLM image sets its own ENTRYPOINT

`vllm/vllm-openai` ships `ENTRYPOINT ["vllm", "serve"]`. Passing `vllm serve <model>` as the
container command runs `vllm serve vllm serve <model>`: argparse takes the literal string `vllm`
as the model argument and reports the real model name as junk.

```
vllm: error: unrecognized arguments: serve nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4
```

That message reads like a bad model name. It is not — the model id is fine, it just arrived in the
wrong argument slot. `serve.sh` passes `--entrypoint vllm` and then `serve <model> --flags`, so the
final command is exactly `vllm serve …` whatever the image's default entrypoint happens to be. If
you write your own `docker run`, check yours first:

```bash
docker inspect vllm/vllm-openai:v0.27.1 \
  --format '{{json .Config.Entrypoint}} {{json .Config.Cmd}}'
```

### The dataset's tool schemas are not OpenAI-compliant

The single most expensive bug in building this sample, because it does not look like a bug.

When2Call's tools come from BFCL and carry two things a strict gateway rejects:

| In the dataset | Why it fails |
| --- | --- |
| `cmd_controller.exe`, `api_token_api.APITokenApi.get_api_tokens` | Function names must match `^[A-Za-z0-9_-]{1,64}$`. Dots are illegal. |
| `"type": "dict"`, `"type": "str"` | JSON Schema wants `object` and `string`. These are Python type names. |

vLLM accepts both. `integrate.api.nvidia.com` does not. So a comparison built on the raw dataset
scores the local endpoint perfectly and returns `400` on every one of 120 examples for hosted
models — **a schema-validation difference wearing the costume of a model result.** Had those
models merely been lenient-but-worse, the published gap would have been about input validation
rather than about the models.

`normalise_row()` repairs both once, before anything is sent, so every endpoint receives identical
valid tools.

### `target_tool` is a spec, not a name

Related, and equally quiet. The dataset stores the gold tool the same way it stores the tool list —
as a JSON string of the whole function spec:

```
returned: find_flights   gold: {"name": "find_flights", "description": "Searches for ...
```

Comparing that against the name a model returns never matches, so **tool-selection accuracy reads
0% for every model** — which looks like a finding about models that cannot pick the right tool,
and is the opposite of the truth. `tool_name_of()` extracts the name from any of the four shapes
the field has taken.

**The general lesson, and why `preflight.py --real` exists:** a probe built from a hand-written,
spec-compliant tool passes on every endpoint and tells you nothing. Send the payload you are
actually going to send.

### Your vLLM release has to know the architecture

Model support ships *in vLLM*, not only in the checkpoint. Serve this model on a release that
predates its support and it fails during config load, ~10 seconds in, before any weights are read:

```
ValidationError: 1 validation error for ModelConfig
  Value error, The checkpoint you are trying to load has model type `nemotron_h`
  but Transformers does not recognize this architecture.
```

The advice in that message — upgrade Transformers, or install it from source — is the generic
Hugging Face text and is **not** the fix here. Two things it does not tell you:

1. **The checkpoint has no remote code.** Its `config.json` declares `"model_type": "nemotron_h"`
   with no `auto_map`, so there is nothing for `--trust-remote-code` to load. The architecture has
   to be compiled into the runtime.
2. **The container's Transformers is not yours to upgrade.** vLLM pins it, and pip-upgrading inside
   a running container gets you a different failure, not a working server.

The fix is to move the *vLLM release* forward. This sample pins `v0.27.1`, which is the release
carrying day-0 support for Nemotron 3.5 Lightning, and the tag the Nemotron team published for it
at release alongside TensorRT-LLM `1.3.0rc24` and SGLang at tip of tree.

`serve.sh` probes for this before starting, so a mismatched image fails in about five seconds with
an explanation rather than five minutes in with a pydantic traceback. The probe asks vLLM's own
config registry rather than comparing version numbers, because a known-good nightly can sort below
a pinned release. `SKIP_PRECHECK=1` bypasses it.

### Picking a vLLM container image for aarch64

This costs people an afternoon, so it is worth stating plainly.

- **`nvcr.io/nvidia/vllm:latest` does not exist.** Pulling it fails with `manifest unknown`.
- **vLLM's `latest` tag is not multi-arch.** Its arm64 builds are published under separate
  arch-suffixed tags — `latest-aarch64`, `nightly-aarch64` — which are single-architecture
  manifests.
- **Versioned tags are multi-arch.** `vllm/vllm-openai:v0.27.1` is a manifest list containing both
  `linux/arm64` (~10.5 GB) and `linux/amd64`, so Docker resolves the correct platform on GB10 with
  no suffix to get wrong.

Check any candidate before committing to it, and check *both* things — architecture and model
support. Getting the architecture right is what lets the container start; it says nothing about
whether that release can load your model:

```bash
docker buildx imagetools inspect vllm/vllm-openai:v0.27.1 | grep -iE "platform|mediatype"
```

A manifest list shows `manifest.list.v2+json` and one `Platform:` line per architecture. A
single-arch image shows only `manifest.v2+json`. Avoid pinning a `nightly-<sha>` tag in anything
you expect others to run — nightly tags are pruned from the registry over time.

- **Hugging Face snapshot directories are symlink farms** into `blobs/`. Mount the whole cache into the container, not just the snapshot directory, or every link dangles. `serve.sh` handles this.
- **JupyterLab inherits group membership at login.** After `sudo gpasswd -a $(whoami) docker` you must restart the JupyterLab *server*; a kernel restart is not sufficient. Use `$(whoami)` rather than `$USER`, which can be `root` in a Jupyter-spawned shell.
- **vLLM 0.26 and later expose the reasoning trace as `reasoning`, not `reasoning_content`.** The notebook handles both.

### `message.content` comes back `None`

The single most confusing thing about serving a reasoning model, and it does not look like a
token-budget problem when it happens:

```
None
  tokens        37 in / 200 out
```

Thinking is **on by default** — the model's chat template sets `enable_thinking` to `True` unless
you say otherwise — and **reasoning and answer share one `max_tokens` budget**. With the budget
set low the model spends all of it reasoning, never reaches its answer, and returns `content:
None` with `finish_reason: "length"`. The tell is `completion_tokens` landing exactly on your
`max_tokens`.

Two fixes, and which one you want depends on the request:

```python
# 1. Give it room, and read the trace off its own field
resp = client.chat.completions.create(..., max_tokens=1536)
msg = resp.choices[0].message
reasoning = getattr(msg, "reasoning", None) or getattr(msg, "reasoning_content", None)

# 2. Or switch thinking off for requests where you just want the answer
resp = client.chat.completions.create(
    ..., extra_body={"chat_template_kwargs": {"enable_thinking": False}})
```

Median completion with thinking on is around 768 tokens, with a tail to 4096 — so 1536 is a
reasonable floor for interactive use and `run_benchmark.py` defaults to 3072.

**Check `finish_reason` before you conclude anything about a model's behaviour.** A truncated
attempt and a considered decision to stay silent look identical if you only read `content`. This
is the same trap described under *Reading results honestly* for the benchmark, where a truncated
reasoning trace scores exactly like a decision not to call a tool.

## Tests

Run `preflight.py` before any sweep. One request per endpoint, and it never prints a key:

```bash
source .venv/bin/activate
./preflight.py                    # all five
./preflight.py --only kimi-k3     # just one
```

It catches the three failures that would otherwise cost you a 600-request run — and two of
them produce a plausible-looking number rather than an error, which is worse:

| Finding | Why it matters |
| --- | --- |
| `401` | Keys are not portable between the public catalog and the internal Hub |
| model not in the catalogue | The configured id is wrong; near-matches are suggested |
| no structured `tool_calls` | Every When2Call example scores as "decided not to call" — a broken measurement that looks like a low score |
| budget exhausted, `finish: length` | Reasoning consumed the budget before the call; recall is understated and tool selection flattered |
| little headroom | Passed, but the harder examples will truncate |

Exit status is non-zero if any endpoint would produce meaningless data, so it drops into CI.

Then verify the rest manually:

```bash
# 1. Endpoint health
curl -s http://localhost:8000/v1/models | python3 -m json.tool

# 1b. Benchmark harness on a small slice
./run_benchmark.py --only spark --n 2

# 2. Tool implementations
python3 -c "
from demo_tools import search_flights, convert_currency
f = search_flights('SFO','AUS','2026-09-14')
cheapest = min(f['results'], key=lambda x: x['fare_usd'])
print(cheapest)
print(convert_currency(cheapest['fare_usd'],'USD','EUR'))
"

# 3. Notebook, top to bottom
jupyter nbconvert --to notebook --execute demo.ipynb --output /tmp/demo-executed.ipynb
```

All three should complete without error. Step 3 is the meaningful one: it exercises the endpoint, the SDK path, and the full agent loop.

## Going Further

Same machine, same 128 GB:

- [Quantize models to NVFP4 with NVIDIA Model Optimizer](https://build.nvidia.com/spark/nvfp4-quantization) — approximately 3.5x memory reduction versus FP16
- [Fine-tune with PyTorch](https://build.nvidia.com/spark/pytorch-fine-tune) — FSDP and LoRA, up to 70B across two DGX Spark systems
- [Unsloth](https://build.nvidia.com/spark/unsloth) · [LLaMA-Factory](https://build.nvidia.com/spark/llama-factory)
- [Connect two DGX Spark systems](https://build.nvidia.com/spark/connect-two-sparks) for 256 GB of pooled memory

Fine-tuning is memory-bound. Full fine-tuning of Llama 3.2 3B, LoRA on Llama 3.1 8B, and QLoRA on Llama 3.3 70B all run here, and none of them fit on a 32 GB consumer GPU.

## Contributing This Sample

Follow the repository contribution flow:

```bash
git checkout main
git pull origin main
git checkout -b feature/nemotron-lightning-vllm-endpoint
git add dgx-spark-samples README.md
git commit -s -m "Add DGX Spark Nemotron 3.5 Lightning vLLM endpoint sample"
git push origin feature/nemotron-lightning-vllm-endpoint
```

Open a pull request to `NVIDIA/nvidia-oci-samples:main` and tag the maintainers listed in the root `MAINTAINERS.md`.

## References

- [Model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4) · [build.nvidia.com](https://build.nvidia.com/nvidia/nemotron-3.5-lightning-30b-a3b)
- [Day-0 support for Nemotron 3.5 Lightning on vLLM](https://vllm.ai/blog/2026-08-10-nemotron-3-5-lightning-vllm) — the source for `v0.27.1` and the tuned DGX Spark profile
- [Nemotron 3.5 Lightning vLLM cookbook](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/usage-cookbook/Nemotron-3.5-Lightning/vllm_cookbook.ipynb)
- [vLLM on DGX Spark playbook](https://build.nvidia.com/spark/vllm)
- [DGX Spark playbooks](https://build.nvidia.com/spark)
- [nvidia/When2Call](https://huggingface.co/datasets/nvidia/When2Call) — agentic tool-calling benchmark (CC-BY-4.0)
