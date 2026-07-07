// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import test from "node:test";
import assert from "node:assert/strict";
import { chatCompletionsUrl, flattenParseEvidence, mapOmniReceiptJson, parseMaybeJson } from "../server/nemotronClient.ts";

test("normalizes chat completion URLs", () => {
  assert.equal(chatCompletionsUrl("https://integrate.api.nvidia.com"), "https://integrate.api.nvidia.com/v1/chat/completions");
  assert.equal(chatCompletionsUrl("https://example.com/v1"), "https://example.com/v1/chat/completions");
});

test("extracts embedded JSON from fenced model output", () => {
  const parsed = parseMaybeJson("```json\n{\"fields\":{\"merchant\":\"ABC\"}}\n```") as Record<string, unknown>;
  assert.deepEqual(parsed, { fields: { merchant: "ABC" } });
});

test("flattens nested parse evidence", () => {
  assert.equal(flattenParseEvidence({ pages: [{ text: "A" }, { markdown: "B" }] }), "A\nB");
});

test("maps Omni JSON aliases", () => {
  const mapped = mapOmniReceiptJson({ fields: { transaction_date: "2026-05-13", payment_method: "Visa", amount: "9.75" }, confidence: { transaction_date: 0.9 } }, "/uploads/x.png");
  assert.equal(mapped.fields.transactionDate, "2026-05-13");
  assert.equal(mapped.fields.paymentMethod, "Visa");
  assert.equal(mapped.fields.amount, 9.75);
  assert.equal(mapped.fields.receiptFileRef, "/uploads/x.png");
});
