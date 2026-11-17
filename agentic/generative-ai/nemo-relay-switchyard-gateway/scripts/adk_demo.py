#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""A Google ADK agent with tools whose model calls are routed by Switchyard through the gateway.

The agent inspects this sample directory with two read-only tools. Its model is LiteLLM's
OpenAI-compatible client pointed at the Relay gateway with the route id `oci/auto` as the model
name, so every model call is routed by Switchyard's stage router and priced by the gateway.
Relay's in-process ADK plugin records the agent, tool, and model scopes in logs/adk-events.jsonl
and propagates its trace to the gateway, which records the routed calls in logs/events.jsonl.

Requires: pip install -r requirements-adk.txt, and scripts/gateway.sh start.
"""
import argparse
import asyncio
import os
from pathlib import Path

from google.adk.agents import LlmAgent
from google.adk.apps import App
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

import nemo_relay
from nemo_relay.integrations.google_adk import NemoRelayPlugin

SAMPLE_ROOT = Path(__file__).resolve().parent.parent
INSPECT_ROOT = Path(os.environ.get("INSPECT_ROOT", SAMPLE_ROOT)).resolve()
QUESTION = (
    "List the files under scripts/, then read relay/plugins.toml.in and report every Switchyard "
    "route id it defines together with the target each route uses."
)


def _inside(path: str) -> Path:
    resolved = (INSPECT_ROOT / path).resolve()
    if resolved != INSPECT_ROOT and INSPECT_ROOT not in resolved.parents:
        raise ValueError("path is outside the inspected directory")
    return resolved


def list_directory(path: str = ".") -> dict:
    """List the entries of a directory inside the inspected project. `path` is relative."""
    target = _inside(path)
    entries = sorted(e.name + ("/" if e.is_dir() else "") for e in target.iterdir() if not e.name.startswith((".", "__")))
    return {"path": path, "entries": entries}


def read_file(path: str, max_chars: int = 12000) -> dict:
    """Read a UTF-8 text file inside the inspected project. `path` is relative; output is truncated to `max_chars`."""
    target = _inside(path)
    text = target.read_text(encoding="utf-8", errors="replace")
    return {"path": path, "chars": len(text), "truncated": len(text) > max_chars, "text": text[:max_chars]}


async def run(route: str, gateway: str, question: str) -> None:
    agent = LlmAgent(
        name="repo_inspector",
        model=LiteLlm(model=f"openai/{route}", api_base=gateway, api_key="relay-routed"),
        instruction=(
            "You inspect a project directory with the tools. Use list_directory and read_file; "
            "never guess file contents. Keep the final answer under 150 words."
        ),
        tools=[list_directory, read_file],
    )
    app = App(name="repo_inspector_app", root_agent=agent, plugins=[NemoRelayPlugin()])
    config = str(SAMPLE_ROOT / "relay" / "plugins-inprocess.toml")
    async with nemo_relay.plugin.activate({}, config):
        try:
            async with Runner(app=app, session_service=InMemorySessionService()) as runner:
                session = await runner.session_service.create_session(app_name=app.name, user_id="user-1")
                with nemo_relay.scope.scope("adk-request", nemo_relay.ScopeType.Agent):
                    async for event in runner.run_async(
                        user_id=session.user_id, session_id=session.id, new_message=types.UserContent(question)
                    ):
                        if event.is_final_response() and event.content and event.content.parts:
                            for part in event.content.parts:
                                if part.text and not part.thought:
                                    print(part.text)
        finally:
            await nemo_relay.subscribers.flush_async()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--route", default=os.environ.get("ADK_ROUTE", "oci/auto"), help="Switchyard route id or OCI model id")
    parser.add_argument("--gateway", default=os.environ.get("RELAY_GATEWAY", "http://127.0.0.1:4040/v1"))
    parser.add_argument("--question", default=QUESTION)
    args = parser.parse_args()
    os.chdir(SAMPLE_ROOT)  # the in-process ATOF sink path is relative to the working directory
    asyncio.run(run(args.route, args.gateway, args.question))


if __name__ == "__main__":
    main()
