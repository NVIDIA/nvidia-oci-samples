// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import { readFile } from "node:fs/promises";
import { join } from "node:path";
import type { AuditLogger } from "./audit.ts";
import type { TripRecord } from "./types.ts";
import { newId, nowIso } from "./util.ts";

interface RuntimePolicy {
  allowedOutboundHosts?: string[];
  toolAllowlist?: string[];
}

export class RuntimeControls {
  policy: RuntimePolicy;
  audit: AuditLogger;

  constructor(policy: RuntimePolicy, audit: AuditLogger) {
    this.policy = policy;
    this.audit = audit;
  }

  static async load(rootDir: string, audit: AuditLogger): Promise<RuntimeControls> {
    try {
      const policy = JSON.parse(await readFile(join(rootDir, "config", "runtime-policy.example.json"), "utf8")) as RuntimePolicy;
      return new RuntimeControls(policy, audit);
    } catch {
      return new RuntimeControls({}, audit);
    }
  }

  async authorizeTool(trip: TripRecord | undefined, toolAction: string, summary: string): Promise<void> {
    const allowed = Array.isArray(this.policy.toolAllowlist) && this.policy.toolAllowlist.includes(toolAction);
    const status = allowed ? "authorized" : "blocked";
    if (trip) {
      trip.agentTrace.steps.push({ id: newId("step"), timestamp: nowIso(), toolAction, status, summary });
    }
    await this.audit.write({
      type: "runtime.tool_authorization",
      severity: allowed ? "info" : "error",
      actor: "nemoclaw-style-runtime",
      action: toolAction,
      tripId: trip?.id,
      details: { status, summary },
    });
    if (!allowed) throw new Error(`Runtime policy blocked tool action: ${toolAction}`);
  }

  async assertOutboundAllowed(hostOrUrl: string, tripId?: string): Promise<void> {
    const host = hostOrUrl.startsWith("http") ? new URL(hostOrUrl).hostname : hostOrUrl;
    const allowed = Array.isArray(this.policy.allowedOutboundHosts) && this.policy.allowedOutboundHosts.includes(host);
    await this.audit.write({
      type: "runtime.network_policy",
      severity: allowed ? "info" : "error",
      actor: "nemoclaw-style-runtime",
      action: "network.allowlist",
      tripId,
      details: { host, allowed },
    });
    if (!allowed) throw new Error(`Outbound host is not allowed by runtime policy: ${host}`);
  }
}
