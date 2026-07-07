// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import { createHash, randomUUID } from "node:crypto";
import { extname } from "node:path";
import type { IncomingMessage, ServerResponse } from "node:http";

export function newId(prefix: string): string {
  return `${prefix}_${randomUUID().replaceAll("-", "").slice(0, 16)}`;
}

export function nowIso(): string {
  return new Date().toISOString();
}

export function stableHash(value: string): string {
  return createHash("sha256").update(value).digest("hex").slice(0, 24);
}

export function sanitizeFileName(value: string): string {
  return value.replace(/[^A-Za-z0-9._ -]/g, "_").replace(/\s+/g, "_").slice(0, 120);
}

export function supportedReceiptFile(fileName: string, mimeType = ""): boolean {
  const ext = extname(fileName).toLowerCase();
  return [".png", ".jpg", ".jpeg"].includes(ext) || ["image/png", "image/jpeg"].includes(mimeType);
}

export function mimeFromPath(path: string): string {
  const ext = extname(path).toLowerCase();
  if (ext === ".html") return "text/html; charset=utf-8";
  if (ext === ".css") return "text/css; charset=utf-8";
  if (ext === ".js") return "text/javascript; charset=utf-8";
  if (ext === ".json") return "application/json; charset=utf-8";
  if (ext === ".png") return "image/png";
  if (ext === ".jpg" || ext === ".jpeg") return "image/jpeg";
  if (ext === ".csv") return "text/csv; charset=utf-8";
  return "application/octet-stream";
}

export async function readRequestJson<T>(request: IncomingMessage): Promise<T> {
  const chunks: Buffer[] = [];
  const maxBytes = 25 * 1024 * 1024;
  let totalBytes = 0;
  for await (const chunk of request) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk);
    totalBytes += buffer.length;
    if (totalBytes > maxBytes) throw new Error("Request body too large");
    chunks.push(buffer);
  }
  const text = Buffer.concat(chunks).toString("utf8");
  return (text ? JSON.parse(text) : {}) as T;
}

export function sendJson(response: ServerResponse, status: number, payload: unknown): void {
  const body = JSON.stringify(redactSecrets(payload), null, 2);
  response.writeHead(status, { "content-type": "application/json; charset=utf-8" });
  response.end(body);
}

export function sendText(response: ServerResponse, status: number, text: string, contentType = "text/plain; charset=utf-8"): void {
  response.writeHead(status, { "content-type": contentType });
  response.end(text);
}

export function parseMoney(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) return Math.round(value * 100) / 100;
  if (typeof value !== "string") return null;
  const cleaned = value.replace(/[^0-9.-]/g, "");
  if (!cleaned) return null;
  const parsed = Number(cleaned);
  return Number.isFinite(parsed) ? Math.round(parsed * 100) / 100 : null;
}

export function hasText(value: unknown): value is string {
  return typeof value === "string" && value.trim().length > 0;
}

export function redactSecrets<T>(value: T): T {
  return JSON.parse(
    JSON.stringify(value, (_key, child) => {
      if (typeof child === "string") {
        return child
          .replace(/nvapi-[A-Za-z0-9_-]+/g, "nvapi-REDACTED")
          .replace(/Bearer\s+[A-Za-z0-9._-]+/g, "Bearer REDACTED")
          .replace(/(api[_-]?key=)[A-Za-z0-9._-]+/gi, "$1REDACTED");
      }
      return child;
    }),
  ) as T;
}

export function titleCase(value: string): string {
  return value
    .toLowerCase()
    .split(/\s+/)
    .filter(Boolean)
    .map((part) => (part.length <= 2 && /^[a-z]+$/.test(part) ? part.toUpperCase() : part[0].toUpperCase() + part.slice(1)))
    .join(" ");
}
