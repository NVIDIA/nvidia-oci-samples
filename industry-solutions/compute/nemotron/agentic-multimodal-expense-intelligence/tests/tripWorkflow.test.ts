// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import test from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, mkdir, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { DataStore } from "../server/dataStore.ts";
import { AuditLogger } from "../server/audit.ts";
import { RuntimeControls } from "../server/governance.ts";
import { TripWorkflow } from "../server/tripWorkflow.ts";
import type { PolicyConfig } from "../server/types.ts";

const policy: PolicyConfig = {
  companyName: "ABC Company",
  currency: "USD",
  requiredFields: ["merchant", "transactionDate", "amount", "currency"],
  receiptRequired: true,
  businessPurposeRequired: true,
  meal: { category: "Meals", attendeesRequiredAbove: 75, warnAbove: 100, tipWarnPercent: 25 },
  lodging: { category: "Lodging", stayDatesWarning: true },
  age: { warnIfOlderThanDays: 90 },
  cash: { warnIfCashPayment: true },
  highAmount: { warnThreshold: 500, blockThreshold: 5000 },
};

test("imports multiple receipt files into one trip and exports CSV", async () => {
  const root = await mkdtemp(join(tmpdir(), "expense-intel-"));
  await mkdir(join(root, "data", "sample-receipts"), { recursive: true });
  await writeFile(join(root, "data", "sample-receipts", "manifest.json"), JSON.stringify([
    { fileName: "meal.png", parseEvidence: "GARDEN CAFE\nDATE: 05/13/2026\nTOTAL - PLUS TAX $9.75\nPAID - CARD# ********7113" },
    { fileName: "parking.png", parseEvidence: "SKY HARBOR PARKING\nDATE: 05/13/2026\nAMOUNT DUE $47.52\nPAID VISA CREDIT" },
  ]));
  const store = new DataStore(root);
  await store.ensure();
  const audit = new AuditLogger(store);
  const controls = new RuntimeControls({ toolAllowlist: ["trip.create", "receipt.store_upload", "policy.evaluate", "csv.export"] }, audit);
  const workflow = new TripWorkflow({ store, audit, controls, policy, extractorOptions: { rootDir: root, mode: "local", parseBaseUrl: "https://integrate.api.nvidia.com", parseModel: "parse", omniBaseUrl: "https://integrate.api.nvidia.com", omniModel: "omni" } });
  const trip = await workflow.createTrip({ employeeName: "ABC Employee Name", tripName: "ABC Trip", tripPurpose: "Customer meetings", totalFiles: 2 });
  await workflow.processFile(trip.id, { fileName: "meal.png", mimeType: "image/png", contentBase64: Buffer.from("fake").toString("base64") });
  await workflow.processFile(trip.id, { fileName: "parking.png", mimeType: "image/png", contentBase64: Buffer.from("fake").toString("base64") });
  const completed = await workflow.completeTrip(trip.id);
  assert.equal(completed.expenseIds.length, 2);
  const csv = await workflow.csvForTrip(trip.id);
  assert.match(csv, /GARDEN CAFE/i);
  assert.match(csv, /SKY HARBOR PARKING/i);
});

test("routes obscured amount receipt to human review", async () => {
  const root = await mkdtemp(join(tmpdir(), "expense-intel-review-"));
  await mkdir(join(root, "data", "sample-receipts"), { recursive: true });
  await writeFile(join(root, "data", "sample-receipts", "manifest.json"), JSON.stringify([
    {
      fileName: "05-obscured-parking-human-review.png",
      parseEvidence: `SKY HARBOR PARKING
MERCHANT: SKY HARBOR PARKING
DATE: 05/13/26 08:10 PM
LOCATION: SJC TERMINAL LOT
PARKING CHARGE $44.00
TAX / SURCHARGE OBSCURED
AMOUNT DUE $--.--
PAYMENT CARD APPROVED
NOTE: FINAL PAYABLE AMOUNT IS COVERED BY STAMP AND CANNOT BE READ`,
    },
  ]));
  const store = new DataStore(root);
  await store.ensure();
  const audit = new AuditLogger(store);
  const controls = new RuntimeControls({ toolAllowlist: ["trip.create", "receipt.store_upload", "policy.evaluate"] }, audit);
  const workflow = new TripWorkflow({ store, audit, controls, policy, extractorOptions: { rootDir: root, mode: "local", parseBaseUrl: "https://integrate.api.nvidia.com", parseModel: "parse", omniBaseUrl: "https://integrate.api.nvidia.com", omniModel: "omni" } });
  const trip = await workflow.createTrip({ employeeName: "ABC Employee Name", tripName: "ABC Trip", tripPurpose: "Customer meetings", totalFiles: 1 });
  const { expense } = await workflow.processFile(trip.id, { fileName: "05-obscured-parking-human-review.png", mimeType: "image/png", contentBase64: Buffer.from("fake").toString("base64") });
  const completed = await workflow.completeTrip(trip.id);
  assert.equal(completed.status, "blocked");
  assert.equal(expense?.savedFields.amount, null);
  assert(expense?.policyChecks.some((check) => check.id === "required-amount" && check.decision === "block"));
});

test("does not mark an empty trip ready for approval", async () => {
  const root = await mkdtemp(join(tmpdir(), "expense-empty-trip-"));
  const store = new DataStore(root);
  await store.ensure();
  const audit = new AuditLogger(store);
  const controls = new RuntimeControls({ toolAllowlist: ["trip.create", "human.approve"] }, audit);
  const workflow = new TripWorkflow({ store, audit, controls, policy, extractorOptions: { rootDir: root, mode: "local", parseBaseUrl: "https://integrate.api.nvidia.com", parseModel: "parse", omniBaseUrl: "https://integrate.api.nvidia.com", omniModel: "omni" } });
  const trip = await workflow.createTrip({ employeeName: "ABC Employee Name", tripName: "ABC Trip", tripPurpose: "Customer meetings", totalFiles: 1 });
  await workflow.processFile(trip.id, { fileName: "notes.txt", mimeType: "text/plain", contentBase64: Buffer.from("fake").toString("base64") });
  const completed = await workflow.completeTrip(trip.id);
  assert.equal(completed.status, "needs_info");
  await assert.rejects(() => workflow.approveTrip(trip.id, "ABC Expense Reviewer"), /not ready for approval/i);
});

test("approval requires a non-employee approver", async () => {
  const root = await mkdtemp(join(tmpdir(), "expense-approval-"));
  await mkdir(join(root, "data", "sample-receipts"), { recursive: true });
  await writeFile(join(root, "data", "sample-receipts", "manifest.json"), JSON.stringify([
    { fileName: "meal.png", parseEvidence: "GARDEN CAFE\nDATE: 05/13/2026\nTOTAL - PLUS TAX $9.75\nPAID - CARD# ********7113" },
  ]));
  const store = new DataStore(root);
  await store.ensure();
  const audit = new AuditLogger(store);
  const controls = new RuntimeControls({ toolAllowlist: ["trip.create", "receipt.store_upload", "policy.evaluate", "human.approve"] }, audit);
  const workflow = new TripWorkflow({ store, audit, controls, policy, extractorOptions: { rootDir: root, mode: "local", parseBaseUrl: "https://integrate.api.nvidia.com", parseModel: "parse", omniBaseUrl: "https://integrate.api.nvidia.com", omniModel: "omni" } });
  const trip = await workflow.createTrip({ employeeName: "ABC Employee Name", tripName: "ABC Trip", tripPurpose: "Customer meetings", totalFiles: 1 });
  await workflow.processFile(trip.id, { fileName: "meal.png", mimeType: "image/png", contentBase64: Buffer.from("fake").toString("base64") });
  await workflow.completeTrip(trip.id);
  await assert.rejects(() => workflow.approveTrip(trip.id, "ABC Employee Name"), /different from the employee/i);
  const approved = await workflow.approveTrip(trip.id, "ABC Expense Reviewer");
  assert.equal(approved.status, "approved");
  assert.equal(approved.approvedBy, "ABC Expense Reviewer");
});

test("policy authorization runs before policy evaluation is persisted", async () => {
  const root = await mkdtemp(join(tmpdir(), "expense-policy-gate-"));
  await mkdir(join(root, "data", "sample-receipts"), { recursive: true });
  await writeFile(join(root, "data", "sample-receipts", "manifest.json"), JSON.stringify([
    { fileName: "meal.png", parseEvidence: "GARDEN CAFE\nDATE: 05/13/2026\nTOTAL - PLUS TAX $9.75\nPAID - CARD# ********7113" },
  ]));
  const store = new DataStore(root);
  await store.ensure();
  const audit = new AuditLogger(store);
  const controls = new RuntimeControls({ toolAllowlist: ["trip.create", "receipt.store_upload"] }, audit);
  const workflow = new TripWorkflow({ store, audit, controls, policy, extractorOptions: { rootDir: root, mode: "local", parseBaseUrl: "https://integrate.api.nvidia.com", parseModel: "parse", omniBaseUrl: "https://integrate.api.nvidia.com", omniModel: "omni" } });
  const trip = await workflow.createTrip({ employeeName: "ABC Employee Name", tripName: "ABC Trip", tripPurpose: "Customer meetings", totalFiles: 1 });
  await assert.rejects(() => workflow.processFile(trip.id, { fileName: "meal.png", mimeType: "image/png", contentBase64: Buffer.from("fake").toString("base64") }), /policy\.evaluate/);
  const state = await store.readState();
  assert.equal(state.expenses.length, 0);
  assert.equal(state.trips[0]?.expenseIds.length, 0);
});
