#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Render the NeMo Relay configuration for one OCI region into a sample-local Relay config
# directory (.relay/), install the NeMo Switchyard plugin bundle into it, and validate the
# plugin and the pricing catalog. Safe to re-run.
#
#   OCI_GENAI_REGION        OCI Generative AI region (default: us-chicago-1)
#   SWITCHYARD_PLUGIN_TAG   NeMo-Relay-Plugins release tag (default: switchyard-plugin-0.3.2)
#   CODEX_MODEL             target of the oci/codex route (default: openai.gpt-oss-120b; see README)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=env.sh
. "$HERE/scripts/env.sh"
: "${OCI_GENAI_REGION:=us-chicago-1}"
: "${SWITCHYARD_PLUGIN_TAG:=switchyard-plugin-0.3.2}"
: "${CODEX_MODEL:=openai.gpt-oss-120b}"

command -v nemo-relay >/dev/null || { echo "nemo-relay CLI not found on PATH; install it first (see README)." >&2; exit 1; }
command -v python3 >/dev/null || { echo "python3 is required." >&2; exit 1; }

CFG="$XDG_CONFIG_HOME/nemo-relay"
mkdir -p "$CFG"

sed "s|@OCI_GENAI_REGION@|$OCI_GENAI_REGION|g" "$HERE/relay/config.toml.in" > "$CFG/config.toml"

if ! grep -qs 'switchyard-plugin' "$CFG/plugins.toml"; then
  echo "Installing the Switchyard plugin bundle ($SWITCHYARD_PLUGIN_TAG) into $CFG ..."
  nemo-relay plugins install --user "github:NVIDIA/NeMo-Relay-Plugins@$SWITCHYARD_PLUGIN_TAG" --no-enable
fi

python3 - "$CFG/plugins.toml" "$HERE/relay/plugins.toml.in" "$OCI_GENAI_REGION" "$HERE/pricing/oci-genai-pricing.json" "$CODEX_MODEL" <<'PY'
import sys
path, template, region, catalog, codex_model = sys.argv[1:6]
marker = "# Appended by scripts/setup.sh"
installed = open(path).read()
if marker in installed:
    # Keep only what `nemo-relay plugins install` wrote; drop the previously rendered block.
    head = installed[: installed.index(marker)]
    head = head[: head.rfind("# ----")] if "# ----" in head else head
    installed = head.rstrip() + "\n"
rendered = (open(template).read().replace("@OCI_GENAI_REGION@", region)
            .replace("@PRICING_CATALOG@", catalog).replace("@CODEX_MODEL@", codex_model))
open(path, "w").write(installed.rstrip() + "\n" + rendered)
print(f"Rendered {path} for region {region}; oci/codex -> {codex_model}")
PY

nemo-relay plugins enable nvidia.switchyard
nemo-relay plugins validate nvidia.switchyard
nemo-relay model-pricing validate "$HERE/pricing/oci-genai-pricing.json"
echo
echo "Relay is configured in $CFG"
echo "Next: export OCI_GENAI_API_KEY=... && scripts/gateway.sh start"
