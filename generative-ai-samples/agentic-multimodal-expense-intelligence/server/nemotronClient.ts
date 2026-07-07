// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import { performance } from "node:perf_hooks";
import type { FieldProvenance, FieldReasoning, ReceiptFields } from "./types.ts";

export function chatCompletionsUrl(baseUrl: string): string {
  const base = baseUrl.replace(/\/+$/, "");
  if (base.endsWith("/v1/chat/completions")) return base;
  if (base.endsWith("/v1")) return `${base}/chat/completions`;
  return `${base}/v1/chat/completions`;
}

export function dataUrlFromBase64(mimeType: string, contentBase64: string): string {
  return `data:${mimeType || "image/png"};base64,${contentBase64}`;
}

export class NemotronChatClient {
  baseUrl: string;
  model: string;
  apiKey: string;
  timeoutMs: number;
  fetchImpl: typeof fetch;

  constructor(input: { baseUrl: string; model: string; apiKey: string; timeoutMs?: number; fetchImpl?: typeof fetch }) {
    this.baseUrl = input.baseUrl;
    this.model = input.model;
    this.apiKey = input.apiKey;
    this.timeoutMs = input.timeoutMs ?? 120_000;
    this.fetchImpl = input.fetchImpl ?? fetch;
  }

  async complete(payload: Record<string, unknown>): Promise<{ content: unknown; rawResponse: Record<string, unknown>; latencyMs: number }> {
    const started = performance.now();
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), this.timeoutMs);
    try {
      const response = await this.fetchImpl(chatCompletionsUrl(this.baseUrl), {
        method: "POST",
        headers: {
          "content-type": "application/json",
          accept: "application/json",
          authorization: `Bearer ${this.apiKey}`,
        },
        body: JSON.stringify({ model: this.model, ...payload }),
        signal: controller.signal,
      });
      const text = await response.text();
      if (!response.ok) throw new Error(`Nemotron HTTP ${response.status}: ${text.slice(0, 500)}`);
      const rawResponse = JSON.parse(text) as Record<string, unknown>;
      return { rawResponse, content: extractModelContent(rawResponse), latencyMs: Math.round(performance.now() - started) };
    } finally {
      clearTimeout(timeout);
    }
  }
}

export function extractModelContent(response: Record<string, unknown>): unknown {
  const choices = Array.isArray(response.choices) ? response.choices : [];
  const first = choices[0] as Record<string, unknown> | undefined;
  const message = first?.message as Record<string, unknown> | undefined;
  if (!message) return response;
  const toolCalls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  if (toolCalls.length > 0) {
    const fn = (toolCalls[0] as Record<string, unknown>).function as Record<string, unknown> | undefined;
    const args = fn?.arguments;
    return typeof args === "string" ? parseMaybeJson(args) : (args ?? {});
  }
  const content = message.content;
  if (typeof content === "string") return parseMaybeJson(content);
  if (Array.isArray(content)) return content.map((part) => typeof part === "string" ? part : String((part as Record<string, unknown>).text ?? "")).join("\n");
  return content ?? {};
}

export function parseMaybeJson(value: string): unknown {
  const stripped = stripCodeFence(value).trim();
  try {
    return JSON.parse(stripped);
  } catch {
    const objectStart = stripped.indexOf("{");
    const objectEnd = stripped.lastIndexOf("}");
    if (objectStart >= 0 && objectEnd > objectStart) {
      try { return JSON.parse(stripped.slice(objectStart, objectEnd + 1)); } catch { return stripped; }
    }
    return stripped;
  }
}

export function stripCodeFence(value: string): string {
  const match = value.match(/^\s*```(?:json)?\s*([\s\S]*?)\s*```\s*$/);
  return match ? match[1] : value;
}

export function flattenParseEvidence(content: unknown): string {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) return dedupeLines(content.map(flattenParseEvidence).join("\n"));
  if (content && typeof content === "object") {
    const obj = content as Record<string, unknown>;
    const parts: string[] = [];
    for (const key of ["text", "markdown", "content"]) if (typeof obj[key] === "string") parts.push(String(obj[key]));
    for (const key of ["pages", "items", "blocks"]) if (Array.isArray(obj[key])) parts.push((obj[key] as unknown[]).map(flattenParseEvidence).join("\n"));
    if (parts.length) return dedupeLines(parts.join("\n"));
    return Object.entries(obj).map(([key, child]) => `${key}: ${flattenParseEvidence(child)}`).join("\n");
  }
  return "";
}

export function dedupeLines(value: string): string {
  const seen = new Set<string>();
  return value.split("\n").filter((line) => {
    const normalized = line.trim();
    if (!normalized || seen.has(normalized)) return false;
    seen.add(normalized);
    return true;
  }).join("\n");
}

export function mapOmniReceiptJson(content: unknown, receiptFileRef?: string): { fields: ReceiptFields; confidence: Record<string, number>; provenance: FieldProvenance[]; reasoning: FieldReasoning[] } {
  const parsed = asRecord(typeof content === "string" ? parseMaybeJson(content) : content) ?? {};
  const source = asRecord(parsed.fields) ?? parsed;
  const fields: ReceiptFields = {
    merchant: stringOrNull(source.merchant),
    transactionDate: stringOrNull(source.transactionDate ?? source.transaction_date),
    amount: numberOrNull(source.amount),
    currency: stringOrNull(source.currency),
    tax: numberOrNull(source.tax),
    tip: numberOrNull(source.tip),
    location: stringOrNull(source.location),
    category: stringOrNull(source.category),
    paymentMethod: stringOrNull(source.paymentMethod ?? source.payment_method),
    receiptFileRef,
    checkInDate: stringOrNull(source.checkInDate ?? source.check_in_date),
    checkOutDate: stringOrNull(source.checkOutDate ?? source.check_out_date),
  };
  const confidenceSource = asRecord(parsed.confidence) ?? {};
  const confidence: Record<string, number> = {};
  for (const key of Object.keys(fields)) {
    // receiptFileRef is caller-supplied, not model-extracted, so it must not get a fabricated confidence.
    if (key === "receiptFileRef") continue;
    const raw = confidenceSource[key] ?? confidenceSource[snakeCase(key)];
    if (typeof raw === "number" && Number.isFinite(raw)) confidence[key] = Math.max(0, Math.min(1, raw));
    else if ((fields as Record<string, unknown>)[key] !== undefined && (fields as Record<string, unknown>)[key] !== null) confidence[key] = 0.72;
  }
  const provenance = Array.isArray(parsed.provenance) ? parsed.provenance.filter(isFieldProvenance) : [];
  const reasoning = Array.isArray(parsed.reasoning) ? parsed.reasoning.filter(isFieldReasoning) : [];
  return { fields, confidence, provenance, reasoning };
}

const PROVENANCE_SOURCES = new Set(["nemotron-parse", "nemotron-omni", "schema-guard", "fixture", "human"]);

function isFieldProvenance(value: unknown): value is FieldProvenance {
  const record = asRecord(value);
  return record !== null && typeof record.field === "string" && typeof record.evidence === "string" && typeof record.source === "string" && PROVENANCE_SOURCES.has(record.source);
}

function isFieldReasoning(value: unknown): value is FieldReasoning {
  const record = asRecord(value);
  return record !== null && typeof record.field === "string" && typeof record.summary === "string";
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : null;
}

function stringOrNull(value: unknown): string | null {
  return typeof value === "string" && value.trim() ? value.trim() : null;
}

function numberOrNull(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) return Math.round(value * 100) / 100;
  if (typeof value === "string") {
    const parsed = Number(value.replace(/[^0-9.-]/g, ""));
    return Number.isFinite(parsed) ? Math.round(parsed * 100) / 100 : null;
  }
  return null;
}

function snakeCase(value: string): string {
  return value.replace(/[A-Z]/g, (letter) => `_${letter.toLowerCase()}`);
}
