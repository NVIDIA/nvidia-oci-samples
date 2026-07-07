// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import test from "node:test";
import assert from "node:assert/strict";
import { RuntimeControls } from "../server/governance.ts";
import type { AuditLogger } from "../server/audit.ts";

function auditSink(): AuditLogger {
  return { write: async () => undefined } as unknown as AuditLogger;
}

test("runtime controls deny tool actions when allowlist is missing", async () => {
  const controls = new RuntimeControls({}, auditSink());
  await assert.rejects(() => controls.authorizeTool(undefined, "trip.create", "Create trip"), /Runtime policy blocked tool action/);
});

test("runtime controls deny outbound hosts when allowlist is missing", async () => {
  const controls = new RuntimeControls({}, auditSink());
  await assert.rejects(() => controls.assertOutboundAllowed("https://integrate.api.nvidia.com"), /Outbound host is not allowed/);
});

test("runtime controls allow configured tool actions and outbound hosts", async () => {
  const controls = new RuntimeControls({ toolAllowlist: ["trip.create"], allowedOutboundHosts: ["integrate.api.nvidia.com"] }, auditSink());
  await controls.authorizeTool(undefined, "trip.create", "Create trip");
  await controls.assertOutboundAllowed("https://integrate.api.nvidia.com/v1/chat/completions");
});
