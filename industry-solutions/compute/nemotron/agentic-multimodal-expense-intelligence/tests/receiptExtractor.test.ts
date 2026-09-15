// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import test from "node:test";
import assert from "node:assert/strict";
import { mkdtemp } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { DataStore } from "../server/dataStore.ts";
import { AuditLogger } from "../server/audit.ts";
import { RuntimeControls } from "../server/governance.ts";
import { extractReceipt } from "../server/receiptExtractor.ts";

test("auto mode fails explicitly when Nemotron keys are missing", async () => {
  const root = await mkdtemp(join(tmpdir(), "expense-no-key-"));
  const store = new DataStore(root);
  await store.ensure();
  const audit = new AuditLogger(store);
  const controls = new RuntimeControls({}, audit);
  await assert.rejects(
    () =>
      extractReceipt(
        {
          fileName: "receipt.png",
          mimeType: "image/png",
          contentBase64: Buffer.from("fake").toString("base64"),
          receiptFileRef: "/uploads/receipt.png",
          tripId: "trip_test",
          expenseId: "exp_test",
        },
        {
          rootDir: root,
          mode: "auto",
          parseBaseUrl: "https://integrate.api.nvidia.com",
          parseModel: "nvidia/nemotron-parse",
          omniBaseUrl: "https://integrate.api.nvidia.com",
          omniModel: "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
        },
        audit,
        controls,
      ),
    /Nemotron Parse\/Omni API key is required/,
  );
  const state = await store.readState();
  assert.equal(state.auditEvents[0]?.severity, "error");
});

test("Nemotron API failures are recorded in audit before bubbling up", async () => {
  const root = await mkdtemp(join(tmpdir(), "expense-api-failure-"));
  const store = new DataStore(root);
  await store.ensure();
  const audit = new AuditLogger(store);
  const controls = new RuntimeControls({ allowedOutboundHosts: ["integrate.api.nvidia.com"] }, audit);
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async () => ({ ok: false, status: 503, text: async () => "service unavailable" }) as Response;
  try {
    await assert.rejects(
      () =>
        extractReceipt(
          {
            fileName: "receipt.png",
            mimeType: "image/png",
            contentBase64: Buffer.from("fake").toString("base64"),
            receiptFileRef: "/uploads/receipt.png",
            tripId: "trip_test",
            expenseId: "exp_test",
          },
          {
            rootDir: root,
            mode: "nemotron",
            parseBaseUrl: "https://integrate.api.nvidia.com",
            parseModel: "nvidia/nemotron-parse",
            parseApiKey: "nvapi-test",
            omniBaseUrl: "https://integrate.api.nvidia.com",
            omniModel: "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
            omniApiKey: "nvapi-test",
          },
          audit,
          controls,
        ),
      /Nemotron HTTP 503/,
    );
  } finally {
    globalThis.fetch = originalFetch;
  }
  const state = await store.readState();
  assert(state.auditEvents.some((event) => event.action === "nemotron.parse.failed" && event.severity === "error"));
});
