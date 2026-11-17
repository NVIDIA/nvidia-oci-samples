#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Run one Codex task through NeMo Relay against OCI Generative AI's Responses API.
# Relay starts a temporary gateway for the run and injects Codex hooks. Codex asks for the
# model `oci/codex`, a Switchyard passthrough route whose target is CODEX_MODEL, so the routing
# decision is recorded and the plugin supplies the OCI key. Requests that match no route
# (such as Codex's model listing) use NEMO_RELAY_OPENAI_AUTH_HEADER.
#
#   OCI_GENAI_API_KEY   OCI Generative AI API key
#   CODEX_MODEL         model id served by OCI's Responses API that supports the Codex tool
#                       loop (see README, "Codex and model choice"); no default. The oci/codex
#                       route is re-rendered when it points at a different model.
#   CODEX_PROMPT        task for Codex (default: write a marker file and reply)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
. "$HERE/scripts/env.sh"
: "${OCI_GENAI_API_KEY:?set OCI_GENAI_API_KEY to an OCI Generative AI API key}"
: "${CODEX_MODEL:?set CODEX_MODEL to a Responses-API model your tenancy serves (see README)}"
: "${CODEX_PROMPT:=Use a file-writing tool to create relay-check.txt in the current working directory containing exactly RELAY_OCI_TOOL_OK. Then reply with RELAY_OCI_RESPONSE_OK.}"
command -v codex >/dev/null || { echo "codex CLI not found on PATH" >&2; exit 1; }

RENDERED="$XDG_CONFIG_HOME/nemo-relay/plugins.toml"
# The id line that follows the codex target table, and nothing else.
RENDERED_TARGET="$(awk '/^\[plugins\.dynamic\.config\.switchyard_config\.targets\.codex\]/{f=1; next} f && /^id = /{gsub(/^id = "|"$/, ""); print; exit}' "$RENDERED" 2>/dev/null || true)"
if [ "$RENDERED_TARGET" != "$CODEX_MODEL" ]; then
  echo "Rendering the oci/codex route for $CODEX_MODEL ..."
  CODEX_MODEL="$CODEX_MODEL" "$HERE/scripts/setup.sh" >/dev/null
fi

export NEMO_RELAY_OPENAI_AUTH_HEADER="Bearer $OCI_GENAI_API_KEY"
# A minimal Codex home keeps your personal plugins and MCP servers out of the run. Codex
# writes session state into its home, so the template is copied under .relay/ first.
export CODEX_HOME="$HERE/.relay/codex-home"
mkdir -p "$CODEX_HOME"
cp "$HERE/codex/config.toml" "$CODEX_HOME/config.toml"
# Codex requires a credential to start; Relay replaces it before the request leaves the gateway.
export OPENAI_API_KEY="${OPENAI_API_KEY:-relay-managed}"

WORKDIR="$HERE/.relay/codex-work"
mkdir -p "$WORKDIR" "$HERE/logs"
rm -f "$WORKDIR/relay-check.txt"
cd "$HERE"
nemo-relay codex -- --model oci/codex exec --skip-git-repo-check --sandbox workspace-write -C "$WORKDIR" "$CODEX_PROMPT" </dev/null
echo
if [ -f "$WORKDIR/relay-check.txt" ]; then
  echo "relay-check.txt: $(cat "$WORKDIR/relay-check.txt")"
fi
echo "Events: $HERE/logs/events.jsonl  (summarize with scripts/summarize_events.py)"
