#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Shared environment for the sample scripts: a sample-local Relay configuration directory
# so nothing is written to your real ~/.config/nemo-relay. Source, do not execute.
# The sample scripts set HERE before sourcing; the fallback serves `source scripts/env.sh` from bash.
SAMPLE_ROOT="${SAMPLE_ROOT:-${HERE:-$(cd "$(dirname "${BASH_SOURCE[0]:-.}")/.." && pwd)}}"
# `nemo-relay plugins install` downloads release assets with the GitHub CLI. Resolve its config
# directory from the caller's environment before XDG_CONFIG_HOME is redirected below.
export GH_CONFIG_DIR="${GH_CONFIG_DIR:-${XDG_CONFIG_HOME:-$HOME/.config}/gh}"
export XDG_CONFIG_HOME="$SAMPLE_ROOT/.relay/config"
export XDG_DATA_HOME="$SAMPLE_ROOT/.relay/data"
export RELAY_BIND="${RELAY_BIND:-127.0.0.1:4040}"
