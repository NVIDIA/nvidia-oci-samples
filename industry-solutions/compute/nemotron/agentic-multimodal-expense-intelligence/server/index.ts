// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import { createServer, type IncomingMessage } from "node:http";
import { readFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import { join, normalize, resolve, sep } from "node:path";
import { DataStore } from "./dataStore.ts";
import { AuditLogger } from "./audit.ts";
import { RuntimeControls } from "./governance.ts";
import { TripWorkflow } from "./tripWorkflow.ts";
import type { PolicyConfig, ReceiptFields } from "./types.ts";
import type { ModelApiKeys } from "./receiptExtractor.ts";
import { mimeFromPath, readRequestJson, sendJson, sendText } from "./util.ts";

const rootDir = process.cwd();
const host = process.env.APP_HOST || "127.0.0.1";
const port = Number(process.env.APP_PORT || 8790);

const store = new DataStore(rootDir);
await store.ensure();
if (process.argv.includes("--reset-only")) {
  await store.reset();
  console.log("Demo state reset.");
  process.exit(0);
}
const audit = new AuditLogger(store);
const controls = await RuntimeControls.load(rootDir, audit);
const policy = JSON.parse(await readFile(join(rootDir, "config", "abc-company-policy.rules.json"), "utf8")) as PolicyConfig;
const mode = (process.env.MODEL_EXECUTION_MODE || "auto") as "auto" | "local" | "nemotron";
const sharedKey = process.env.NVIDIA_API_KEY || "";
const workflow = new TripWorkflow({
  store,
  audit,
  controls,
  policy,
  extractorOptions: {
    rootDir,
    mode,
    parseBaseUrl: process.env.NEMOTRON_PARSE_BASE_URL || "https://integrate.api.nvidia.com",
    parseModel: process.env.NEMOTRON_PARSE_MODEL || "nvidia/nemotron-parse",
    parseApiKey: process.env.NEMOTRON_PARSE_API_KEY || sharedKey,
    omniBaseUrl: process.env.NEMOTRON_OMNI_BASE_URL || "https://integrate.api.nvidia.com",
    omniModel: process.env.NEMOTRON_OMNI_MODEL || "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
    omniApiKey: process.env.NEMOTRON_OMNI_API_KEY || sharedKey,
  },
});

const server = createServer(async (request, response) => {
  try {
    const url = new URL(request.url || "/", `http://${request.headers.host || `${host}:${port}`}`);
    if (request.method === "GET" && url.pathname === "/api/health") return sendJson(response, 200, {
      ok: true,
      modelExecutionMode: mode,
      parseBaseUrl: process.env.NEMOTRON_PARSE_BASE_URL || "https://integrate.api.nvidia.com",
      omniBaseUrl: process.env.NEMOTRON_OMNI_BASE_URL || "https://integrate.api.nvidia.com",
      parseModel: process.env.NEMOTRON_PARSE_MODEL || "nvidia/nemotron-parse",
      omniModel: process.env.NEMOTRON_OMNI_MODEL || "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
      serverEnvKeyConfigured: Boolean(sharedKey || process.env.NEMOTRON_PARSE_API_KEY || process.env.NEMOTRON_OMNI_API_KEY),
    });
    if (request.method === "GET" && url.pathname === "/api/state") return sendJson(response, 200, await workflow.viewModel());
    if (request.method === "POST" && url.pathname === "/api/reset") {
      await store.reset();
      return sendJson(response, 200, { ok: true });
    }
    if (request.method === "POST" && url.pathname === "/api/trips") {
      const body = await readRequestJson<{ employeeName: string; tripName: string; tripPurpose: string; totalFiles: number }>(request);
      if (typeof body.employeeName !== "string" || typeof body.tripName !== "string" || typeof body.tripPurpose !== "string") {
        return sendJson(response, 400, { error: "employeeName, tripName, and tripPurpose must be provided as strings." });
      }
      const trip = await workflow.createTrip(body);
      return sendJson(response, 201, { trip });
    }
    const processMatch = url.pathname.match(/^\/api\/trips\/([^/]+)\/files$/);
    if (request.method === "POST" && processMatch) {
      const body = await readRequestJson<{ fileName: string; mimeType: string; contentBase64: string; lastModified?: number; modelApiKeys?: ModelApiKeys }>(request);
      return sendJson(response, 200, await workflow.processFile(processMatch[1], body));
    }
    const completeMatch = url.pathname.match(/^\/api\/trips\/([^/]+)\/complete$/);
    if (request.method === "POST" && completeMatch) return sendJson(response, 200, { trip: await workflow.completeTrip(completeMatch[1]) });
    const approveMatch = url.pathname.match(/^\/api\/trips\/([^/]+)\/approve$/);
    if (request.method === "POST" && approveMatch) {
      const body = await readRequestJson<{ approvedBy: string }>(request);
      if (typeof body.approvedBy !== "string" || body.approvedBy.trim() === "") {
        return sendJson(response, 400, { error: "approvedBy is required." });
      }
      return sendJson(response, 200, { trip: await workflow.approveTrip(approveMatch[1], body.approvedBy) });
    }
    const csvMatch = url.pathname.match(/^\/api\/trips\/([^/]+)\/export\.csv$/);
    if (request.method === "GET" && csvMatch) return sendText(response, 200, await workflow.csvForTrip(csvMatch[1]), "text/csv; charset=utf-8");
    const fieldMatch = url.pathname.match(/^\/api\/expenses\/([^/]+)\/fields$/);
    if (request.method === "PATCH" && fieldMatch) {
      const body = await readRequestJson<{ fields: ReceiptFields }>(request);
      return sendJson(response, 200, await workflow.saveExpenseFields(fieldMatch[1], body.fields));
    }
    return serveStatic(url.pathname, request, response);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return sendJson(response, 500, { error: message });
  }
});

server.on("error", (error: NodeJS.ErrnoException) => {
  if (error.code === "EADDRINUSE") {
    console.error(`Port ${port} is already in use. Set APP_PORT to a free port and restart.`);
  } else {
    console.error(`HTTP server error: ${error.message}`);
  }
  process.exit(1);
});

server.listen(port, host, () => {
  if (!isLoopbackHost(host)) {
    console.warn(`APP_HOST is set to "${host}", which is not a loopback address. This sample has no authentication; endpoints like POST /api/reset and the trip/expense mutation routes would be open to any caller on a shared network.`);
  }
  console.log(`Agentic Multimodal Expense Intelligence running at http://${host}:${port}/`);
});

function isLoopbackHost(value: string): boolean {
  return value === "127.0.0.1" || value === "::1" || value.toLowerCase() === "localhost";
}

async function serveStatic(pathname: string, request: IncomingMessage, response: import("node:http").ServerResponse): Promise<void> {
  const publicRoot = resolve(rootDir, "public");
  const uploadsRoot = resolve(rootDir, "data", "uploads");
  let candidate: string;
  if (pathname === "/") candidate = resolve(publicRoot, "index.html");
  else if (pathname.startsWith("/uploads/")) {
    if (!isLocalRequest(request)) return sendText(response, 403, "Uploaded receipts are served only to local demo clients.");
    candidate = resolve(uploadsRoot, normalize(pathname).replace(/^\/uploads\/?/, ""));
  } else {
    candidate = resolve(publicRoot, normalize(pathname).replace(/^\/+/, ""));
  }
  if (!insideRoot(candidate, publicRoot) && !insideRoot(candidate, uploadsRoot)) return sendText(response, 403, "Forbidden");
  if (!existsSync(candidate)) return sendText(response, 404, "Not found");
  response.writeHead(200, { "content-type": mimeFromPath(candidate) });
  response.end(await readFile(candidate));
}

function insideRoot(candidate: string, allowedRoot: string): boolean {
  return candidate === allowedRoot || candidate.startsWith(`${allowedRoot}${sep}`);
}

function isLocalRequest(request: IncomingMessage): boolean {
  const address = request.socket.remoteAddress ?? "";
  return address === "127.0.0.1" || address === "::1" || address === "::ffff:127.0.0.1";
}
