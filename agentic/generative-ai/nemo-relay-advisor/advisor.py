# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Nemotron deployment advisor: a LangChain agent on OCI Generative AI, observed by NeMo Relay.

The agent is LangChain's ``create_agent`` with ``ChatOCIGenAI`` (managed OCI Generative AI) as the
model. NVIDIA NeMo Relay records and governs every model call and tool call through one line of
middleware, ``NemoRelayMiddleware()``, and exports the whole run as a portable ATIF trajectory. The
agent code is unchanged from a normal LangChain agent; Relay only observes it.

The agent answers a real question ("can we run Nemotron on this tenancy?") by calling two read-only
tools that query the tenancy: the managed Generative AI model catalog and the GPU service limits. It
then recommends either a managed catalog model or self-hosting on OKE.

Everything is parameterized through environment variables; no account identifiers are hard-coded.
"""

import json
import os
import subprocess

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_oci.chat_models import ChatOCIGenAI

import nemo_relay
from nemo_relay.integrations.langchain import NemoRelayMiddleware

# --- configuration (all from the environment) -------------------------------------------------
REGION = os.environ.get("OCI_REGION", "us-chicago-1")            # a region where OCI Generative AI runs
COMPARTMENT = os.environ["OCI_COMPARTMENT_ID"]                    # required: your compartment (or tenancy) OCID
MODEL = os.environ.get("ADVISOR_MODEL", "meta.llama-3.3-70b-instruct")  # any managed chat model in the catalog
PROFILE = os.environ.get("OCI_PROFILE", "DEFAULT")               # a profile in ~/.oci/config
# Compute capacity reports must be requested against the root (tenancy) compartment. If
# OCI_COMPARTMENT_ID is a child compartment, set OCI_TENANCY_ID to your tenancy OCID.
TENANCY = os.environ.get("OCI_TENANCY_ID", COMPARTMENT)
# API_KEY, SECURITY_TOKEN, INSTANCE_PRINCIPAL, or RESOURCE_PRINCIPAL.
AUTH_TYPE = os.environ.get("OCI_AUTH_TYPE", "API_KEY").upper()

# Map the SDK auth type onto the matching `oci` CLI flags for the read-only tool calls below.
_CLI_AUTH = ["--auth", AUTH_TYPE.lower()]
if AUTH_TYPE in ("API_KEY", "SECURITY_TOKEN"):
    _CLI_AUTH += ["--profile", PROFILE]


def _oci_cli(*args: str) -> str:
    """Run a read-only `oci` CLI query and return its stdout (or a short error string)."""
    try:
        result = subprocess.run(
            ["oci", *args, *_CLI_AUTH], capture_output=True, text=True, timeout=90, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return json.dumps({"error": str(exc)[-300:]})
    return result.stdout if result.returncode == 0 else json.dumps(
        {"error": result.stderr.strip()[-300:]}
    )


@tool
def list_genai_models(keyword: str) -> str:
    """Search the managed OCI Generative AI model catalog by keyword (e.g. 'nemotron', 'llama')."""
    print(f"  [tool] list_genai_models(keyword={keyword!r})")
    raw = _oci_cli(
        "generative-ai", "model-collection", "list-models",
        "--compartment-id", COMPARTMENT, "--region", REGION,
        "--query", 'data.items[*]."display-name"',
    )
    try:
        payload = json.loads(raw)
    except (ValueError, TypeError):
        return raw[:300]
    # A successful catalog query returns a JSON array of display-name strings; anything else
    # (for example the {"error": ...} object from _oci_cli) means the call did not succeed, so
    # surface it rather than reporting an empty catalog.
    if not isinstance(payload, list) or not all(isinstance(n, str) for n in payload):
        return raw[:300]
    names = sorted({n for n in payload if keyword.lower() in n.lower()})
    return json.dumps({"region": REGION, "keyword": keyword, "matches": names[:40]})


@tool
def gpu_capacity(shape: str) -> str:
    """Check real host capacity for a GPU shape (e.g. 'VM.GPU.A10.2') per availability domain.

    Uses an OCI Compute capacity report, the authoritative signal for whether a shape can actually
    be launched now (``AVAILABLE`` vs ``OUT_OF_HOST_CAPACITY``), not a service-limit value. On-demand
    capacity is a point-in-time reading and is not guaranteed at launch.
    """
    print(f"  [tool] gpu_capacity(shape={shape!r})")
    ad_raw = _oci_cli(
        "iam", "availability-domain", "list", "--compartment-id", TENANCY,
        "--region", REGION, "--query", "data[*].name",
    )
    try:
        ads = json.loads(ad_raw)
    except (ValueError, TypeError):
        return ad_raw[:300]
    if not isinstance(ads, list) or not all(isinstance(a, str) for a in ads):
        return ad_raw[:300]
    rows = []
    for ad in ads[:10]:
        raw = _oci_cli(
            "compute", "compute-capacity-report", "create", "--compartment-id", TENANCY,
            "--region", REGION, "--availability-domain", ad,
            "--shape-availabilities", json.dumps([{"instanceShape": shape}]),
            "--query", 'data."shape-availabilities"[0].{status:"availability-status",available:"available-count"}',
        )
        row = {"ad": ad, "shape": shape}
        try:
            info = json.loads(raw)
        except (ValueError, TypeError):
            info = {"error": "could not parse capacity report", "raw": raw[:120]}
        row.update(info if isinstance(info, dict) else {"error": "unexpected capacity report shape", "raw": raw[:120]})
        rows.append(row)
    return json.dumps({"region": REGION, "shape": shape, "by_ad": rows})


QUESTION = (
    "We want to run NVIDIA Nemotron on this OCI tenancy (requested by jane.doe@example.com). "
    "First check which Nemotron or Llama models the managed OCI Generative AI service offers here, "
    "then check whether the VM.GPU.A10.2 shape has host capacity to self-host on OKE. "
    "Finish with a concrete deployment recommendation."
)

SYSTEM = (
    "You are an Oracle Cloud deployment advisor. Gather real data with the tools before "
    "answering: search the model catalog for 'nemotron' AND for 'llama', and check host "
    "capacity for the VM.GPU.A10.2 shape. The two deployment paths are: (a) use a model "
    "from the managed Generative AI catalog as-is, or (b) self-host an open-weights model "
    "such as NVIDIA Nemotron on OKE using the tenancy's own GPU capacity. The capacity "
    "check is a point-in-time Compute capacity report (AVAILABLE vs OUT_OF_HOST_CAPACITY), "
    "not a guarantee of capacity at launch. In the final recommendation (under 150 words), cite the "
    "exact catalog results and the per-AD capacity status you found. If the managed catalog has no "
    "Nemotron model AND no availability domain has host capacity for the shape, neither path can run "
    "Nemotron right now: say so plainly and give next steps (try another region or shape, or request "
    "GPU capacity), and only mention a managed non-Nemotron model as an alternative while making clear "
    "it is not Nemotron. Otherwise pick the path that runs Nemotron."
)


class _OCIChatCompat(ChatOCIGenAI):
    """ChatOCIGenAI that drops HTTP-style ``extra_headers`` kwargs.

    Only needed on ``nemo-relay < 0.9.0``: that middleware forwarded Relay request headers as
    ``model_settings["extra_headers"]`` for models without ``default_headers``, and the OCI SDK
    rejects unknown request fields. NeMo Relay 0.9.0 no longer injects them for OCI models
    (NVIDIA/NeMo-Relay#1012), so on 0.9.0+ you can use ``ChatOCIGenAI`` directly and delete this class.
    """

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        kwargs.pop("extra_headers", None)
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def main() -> None:
    llm = _OCIChatCompat(
        model_id=MODEL,
        service_endpoint=f"https://inference.generativeai.{REGION}.oci.oraclecloud.com",
        compartment_id=COMPARTMENT,
        auth_type=AUTH_TYPE,
        auth_profile=PROFILE,
        model_kwargs={"temperature": 0.0, "max_tokens": 600},
    )

    # The only NeMo Relay line the agent needs: one middleware entry. Everything else is a
    # standard LangChain agent.
    agent = create_agent(
        model=llm,
        tools=[list_genai_models, gpu_capacity],
        middleware=[NemoRelayMiddleware()],
        system_prompt=SYSTEM,
    )

    exporter = nemo_relay.AtifExporter(
        "nemotron-advisor-langchain", "nemotron-deployment-advisor", "2.0.0",
        model_name=MODEL,
    )
    exporter.register("atif_lc_advisor")

    with nemo_relay.scope.scope("nemotron-advisor", nemo_relay.ScopeType.Agent):
        result = agent.invoke({"messages": [{"role": "user", "content": QUESTION}]})

    final = result["messages"][-1]
    print("\n=== FINAL ANSWER ===\n" + (final.content or ""))

    traj = exporter.export()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "advisor-trajectory.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(traj, f, indent=2)
    print(f"\nATIF trajectory written to {out}")
    print("final_metrics:", json.dumps(traj.get("final_metrics"), indent=2))
    print("steps:", len(traj.get("steps", [])))


if __name__ == "__main__":
    main()
