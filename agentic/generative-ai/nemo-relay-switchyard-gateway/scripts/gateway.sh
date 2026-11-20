#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Start, stop, or check the standalone NeMo Relay gateway configured by scripts/setup.sh.
# The gateway runs from the sample directory so ATOF events land in logs/events.jsonl.
#
#   OCI_GENAI_API_KEY   OCI Generative AI API key; the Switchyard plugin reads it for routed
#                       calls. Direct calls to a model id send the client's own key.
#   RELAY_BIND          listen address (default 127.0.0.1:4040)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
. "$HERE/scripts/env.sh"
PIDFILE="$HERE/.relay/gateway.pid"

case "${1:-}" in
  start)
    : "${OCI_GENAI_API_KEY:?set OCI_GENAI_API_KEY to an OCI Generative AI API key}"
    if curl -sf --max-time 2 "http://$RELAY_BIND/healthz" >/dev/null 2>&1; then
      echo "a gateway already answers on http://$RELAY_BIND; run '$0 stop' first" >&2
      exit 1
    fi
    mkdir -p "$HERE/logs"
    cd "$HERE"
    nohup nemo-relay --bind "$RELAY_BIND" > "$HERE/logs/gateway.log" 2>&1 &
    GATEWAY_PID=$!
    echo "$GATEWAY_PID" > "$PIDFILE"
    for _ in $(seq 1 80); do
      kill -0 "$GATEWAY_PID" 2>/dev/null || break
      curl -sf --max-time 2 "http://$RELAY_BIND/healthz" >/dev/null 2>&1 && break
      sleep 0.25
    done
    if ! kill -0 "$GATEWAY_PID" 2>/dev/null || ! curl -sf --max-time 2 "http://$RELAY_BIND/healthz" >/dev/null 2>&1; then
      # A child that is alive but never became healthy would otherwise outlive its PID file.
      kill "$GATEWAY_PID" 2>/dev/null || true
      wait "$GATEWAY_PID" 2>/dev/null || true
      rm -f "$PIDFILE"
      echo "gateway did not come up; see logs/gateway.log" >&2
      exit 1
    fi
    echo "NeMo Relay gateway listening on http://$RELAY_BIND (events: logs/events.jsonl, log: logs/gateway.log)"
    ;;
  stop)
    nemo-relay gateway stop --bind "$RELAY_BIND" 2>/dev/null || { [ -f "$PIDFILE" ] && kill "$(cat "$PIDFILE")" 2>/dev/null || true; }
    rm -f "$PIDFILE"
    echo "gateway stopped"
    ;;
  status)
    curl -s --max-time 2 "http://$RELAY_BIND/healthz" && echo
    ;;
  *)
    echo "usage: $0 start|stop|status" >&2
    exit 2
    ;;
esac
