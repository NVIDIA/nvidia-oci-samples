// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import { mkdir, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { performance } from "node:perf_hooks";
import type { AuditLogger } from "./audit.ts";
import type { DataStore } from "./dataStore.ts";
import type { RuntimeControls } from "./governance.ts";
import type { ExpenseRecord, PolicyConfig, ReceiptFields, TripRecord } from "./types.ts";
import { evaluatePolicy, statusFromChecks } from "./policyEngine.ts";
import { tripCsv } from "./csvExport.ts";
import { extractReceipt, type ExtractorOptions, type ModelApiKeys, withRequestApiKeys } from "./receiptExtractor.ts";
import { newId, nowIso, sanitizeFileName, stableHash, supportedReceiptFile } from "./util.ts";

export class TripWorkflow {
  store: DataStore;
  audit: AuditLogger;
  controls: RuntimeControls;
  policy: PolicyConfig;
  extractorOptions: ExtractorOptions;

  constructor(input: { store: DataStore; audit: AuditLogger; controls: RuntimeControls; policy: PolicyConfig; extractorOptions: ExtractorOptions }) {
    this.store = input.store;
    this.audit = input.audit;
    this.controls = input.controls;
    this.policy = input.policy;
    this.extractorOptions = input.extractorOptions;
  }

  async createTrip(input: { employeeName: string; tripName: string; tripPurpose: string; totalFiles: number }): Promise<TripRecord> {
    const now = nowIso();
    const trip: TripRecord = {
      id: newId("trip"),
      employeeName: input.employeeName.trim() || "ABC Employee Name",
      tripName: input.tripName.trim() || "ABC Company Trip",
      tripPurpose: input.tripPurpose.trim(),
      status: "importing",
      createdAt: now,
      updatedAt: now,
      totalFiles: input.totalFiles,
      processedFiles: 0,
      skippedFiles: [],
      expenseIds: [],
      agentTrace: { agentRunId: newId("agent"), runtime: "local-agent", runtimeControl: "nemoclaw-style-policy", steps: [] },
    };
    await this.controls.authorizeTool(trip, "trip.create", "Create one trip report to group multiple receipt images.");
    await this.store.addTrip(trip);
    await this.audit.write({ type: "trip.created", actor: "expense-intelligence-agent", action: "trip.create", tripId: trip.id, details: { employeeName: trip.employeeName, totalFiles: trip.totalFiles } });
    return trip;
  }

  async processFile(tripId: string, file: { fileName: string; mimeType: string; contentBase64: string; lastModified?: number; modelApiKeys?: ModelApiKeys }): Promise<{ trip: TripRecord; expense?: ExpenseRecord; skipped?: { fileName: string; reason: string } }> {
    const trip = await this.mustTrip(tripId);
    const started = performance.now();
    if (!supportedReceiptFile(file.fileName, file.mimeType)) {
      const skipped = { fileName: file.fileName, reason: "Unsupported receipt file type. Use JPG, JPEG, or PNG." };
      trip.skippedFiles.push(skipped);
      trip.processedFiles += 1;
      trip.updatedAt = nowIso();
      await this.store.updateTrip(trip);
      await this.audit.write({ type: "receipt.skipped", severity: "warn", actor: "expense-intelligence-agent", action: "receipt.filter", tripId, details: skipped });
      return { trip, skipped };
    }

    await this.controls.authorizeTool(trip, "receipt.store_upload", `Store uploaded receipt ${file.fileName}.`);
    const safeName = `${trip.id}-${stableHash(file.fileName + String(file.lastModified ?? ""))}-${sanitizeFileName(file.fileName)}`;
    const uploadPath = join(this.store.uploadsDir, safeName);
    await mkdir(this.store.uploadsDir, { recursive: true });
    await writeFile(uploadPath, Buffer.from(file.contentBase64, "base64"));
    const receiptFileRef = `/uploads/${safeName}`;

    const expenseId = newId("exp");
    const extractStarted = performance.now();
    const requestOptions = withRequestApiKeys(this.extractorOptions, file.modelApiKeys);
    await this.audit.write({
      type: "model.routing",
      actor: "expense-intelligence-agent",
      action: "model.route",
      tripId,
      expenseId,
      details: {
        executionMode: requestOptions.mode,
        parseModel: requestOptions.parseModel,
        omniModel: requestOptions.omniModel,
        browserSessionKeyProvided: Boolean(file.modelApiKeys?.nvidiaApiKey || file.modelApiKeys?.parseApiKey || file.modelApiKeys?.omniApiKey),
        serverEnvKeyConfigured: Boolean(this.extractorOptions.parseApiKey && this.extractorOptions.omniApiKey),
      },
    });
    const extracted = await extractReceipt({ fileName: file.fileName, mimeType: file.mimeType, contentBase64: file.contentBase64, receiptFileRef, tripId, expenseId }, requestOptions, this.audit, this.controls);
    const extractionMs = Math.round(performance.now() - extractStarted);
    const fields: ReceiptFields = { ...extracted.fields, receiptFileRef };
    const policyStarted = performance.now();
    await this.controls.authorizeTool(trip, "policy.evaluate", `Evaluate ABC Company policy for ${file.fileName}.`);
    const policyChecks = evaluatePolicy({ savedFields: fields }, trip.tripPurpose, this.policy);
    const policyMs = Math.round(performance.now() - policyStarted);
    const expense: ExpenseRecord = {
      id: expenseId,
      tripId,
      fileName: file.fileName,
      status: statusFromChecks(policyChecks),
      createdAt: nowIso(),
      updatedAt: nowIso(),
      extracted: { ...extracted, fields },
      savedFields: fields,
      policyChecks,
      performance: [
        { name: "receipt_upload", durationMs: Math.max(1, Math.round(extractStarted - started)) },
        { name: "parse_omni_repair", durationMs: extractionMs },
        { name: "abc_policy", durationMs: policyMs },
      ],
    };
    await this.store.addExpense(expense);
    trip.expenseIds.push(expense.id);
    trip.processedFiles += 1;
    trip.updatedAt = nowIso();
    await this.store.updateTrip(trip);
    await this.audit.write({ type: "receipt.processed", actor: "expense-intelligence-agent", action: "receipt.process", tripId, expenseId, details: { fileName: file.fileName, status: expense.status, durationMs: Math.round(performance.now() - started) } });
    return { trip, expense };
  }

  async completeTrip(tripId: string): Promise<TripRecord> {
    const trip = await this.mustTrip(tripId);
    const expenses = await this.store.tripExpenses(tripId);
    trip.status = tripStatus(expenses);
    trip.updatedAt = nowIso();
    await this.store.updateTrip(trip);
    await this.audit.write({ type: "trip.completed", actor: "expense-intelligence-agent", action: "trip.complete", tripId, details: { status: trip.status, expenseCount: expenses.length } });
    return trip;
  }

  async saveExpenseFields(expenseId: string, fields: ReceiptFields): Promise<{ trip: TripRecord; expense: ExpenseRecord }> {
    const expense = await this.mustExpense(expenseId);
    expense.savedFields = { ...expense.savedFields, ...fields };
    expense.policyChecks = evaluatePolicy({ savedFields: expense.savedFields }, (await this.mustTrip(expense.tripId)).tripPurpose, this.policy);
    expense.status = statusFromChecks(expense.policyChecks);
    expense.updatedAt = nowIso();
    await this.store.updateExpense(expense);
    const trip = await this.completeTrip(expense.tripId);
    await this.audit.write({ type: "human.correction", actor: "reviewer", action: "expense.save_fields", tripId: trip.id, expenseId, details: { fields: expense.savedFields } });
    return { trip, expense };
  }

  async approveTrip(tripId: string, approvedBy: string): Promise<TripRecord> {
    const approver = approvedBy.trim();
    if (!approver) throw new Error("Approver identity is required.");
    const trip = await this.completeTrip(tripId);
    await this.controls.authorizeTool(trip, "human.approve", "Human approval gate before downstream expense handoff.");
    if (trip.status !== "ready_for_review") throw new Error(`Trip is not ready for approval. Current status: ${trip.status}`);
    if (samePerson(approver, trip.employeeName)) throw new Error("Approver must be different from the employee submitting the trip.");
    trip.status = "approved";
    trip.approvedAt = nowIso();
    trip.approvedBy = approver;
    trip.updatedAt = nowIso();
    await this.store.updateTrip(trip);
    await this.audit.write({ type: "human.approval", actor: trip.approvedBy, action: "trip.approve", tripId, details: { approvedAt: trip.approvedAt } });
    return trip;
  }

  async csvForTrip(tripId: string): Promise<string> {
    const trip = await this.mustTrip(tripId);
    await this.controls.authorizeTool(trip, "csv.export", "Export saved expense fields as CSV for downstream systems.");
    return tripCsv(trip, await this.store.tripExpenses(tripId));
  }

  async viewModel(): Promise<Record<string, unknown>> {
    const state = await this.store.readState();
    const trips = state.trips.map((trip) => ({ ...trip, expenses: state.expenses.filter((expense) => expense.tripId === trip.id).reverse() }));
    return { trips, auditEvents: state.auditEvents, stats: computeStats(state.expenses) };
  }

  private async mustTrip(id: string): Promise<TripRecord> {
    const trip = await this.store.trip(id);
    if (!trip) throw new Error(`Trip not found: ${id}`);
    return trip;
  }

  private async mustExpense(id: string): Promise<ExpenseRecord> {
    const expense = await this.store.expense(id);
    if (!expense) throw new Error(`Expense not found: ${id}`);
    return expense;
  }
}

function tripStatus(expenses: ExpenseRecord[]): TripRecord["status"] {
  if (expenses.length === 0) return "needs_info";
  if (expenses.some((expense) => expense.status === "blocked")) return "blocked";
  if (expenses.some((expense) => expense.status === "needs_info")) return "needs_info";
  return "ready_for_review";
}

function samePerson(left: string, right: string): boolean {
  return left.trim().localeCompare(right.trim(), undefined, { sensitivity: "base" }) === 0;
}

function computeStats(expenses: ExpenseRecord[]): Record<string, unknown> {
  const fields = expenses.flatMap((expense) => Object.entries(expense.savedFields).map(([field, value]) => ({ field, value, confidence: expense.extracted.confidence[field] })));
  const confidences = fields.map((item) => item.confidence).filter((value): value is number => typeof value === "number" && Number.isFinite(value));
  const totalMs = expenses.reduce((sum, expense) => sum + expense.performance.reduce((inner, phase) => inner + phase.durationMs, 0), 0);
  const phaseTotals: Record<string, number> = {};
  for (const expense of expenses) for (const phase of expense.performance) phaseTotals[phase.name] = (phaseTotals[phase.name] ?? 0) + phase.durationMs;
  return {
    receiptCount: expenses.length,
    readyCount: expenses.filter((expense) => expense.status === "ready_for_review" || expense.status === "approved").length,
    blockedCount: expenses.filter((expense) => expense.status === "blocked").length,
    warningCount: expenses.flatMap((expense) => expense.policyChecks).filter((check) => check.decision === "warn").length,
    averageConfidence: confidences.length ? Math.round((confidences.reduce((sum, value) => sum + value, 0) / confidences.length) * 100) : 0,
    totalProcessingMs: totalMs,
    phaseTotals,
  };
}
