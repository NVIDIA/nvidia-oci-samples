// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

export type ExpenseCategory = "Meals" | "Lodging" | "Transportation" | "Airfare" | "Supplies" | "Other";
export type ExpenseStatus = "needs_info" | "ready_for_review" | "blocked" | "approved";
export type TripStatus = "importing" | "needs_info" | "ready_for_review" | "approved" | "blocked";
export type PolicyDecision = "pass" | "warn" | "block";
export type Severity = "info" | "warn" | "error";

export interface ReceiptFields {
  merchant?: string | null;
  transactionDate?: string | null;
  amount?: number | null;
  currency?: string | null;
  tax?: number | null;
  tip?: number | null;
  location?: string | null;
  category?: ExpenseCategory | string | null;
  paymentMethod?: string | null;
  receiptFileRef?: string | null;
  checkInDate?: string | null;
  checkOutDate?: string | null;
}

export interface FieldProvenance {
  field: string;
  source: "nemotron-parse" | "nemotron-omni" | "schema-guard" | "fixture" | "human";
  evidence: string;
}

export interface FieldReasoning {
  field: string;
  summary: string;
  evidence?: string[];
  rawValue?: string | null;
  normalizedValue?: string | null;
  confidence?: number;
}

export interface ExtractionResult {
  provider: "local-fixture" | "nvidia-nemotron";
  model: string;
  fields: ReceiptFields;
  confidence: Record<string, number>;
  provenance: FieldProvenance[];
  reasoning: FieldReasoning[];
  rawText: string;
  parsePasses: Array<{ pass: string; mode: string; latencyMs: number; evidenceChars: number }>;
  schemaVersion: "expense-receipt-v1";
}

export interface PolicyCheck {
  id: string;
  decision: PolicyDecision;
  reason: string;
  evidence?: Record<string, unknown>;
}

export interface AuditEvent {
  id: string;
  timestamp: string;
  type: string;
  severity: Severity;
  actor: string;
  action: string;
  tripId?: string;
  expenseId?: string;
  details?: Record<string, unknown>;
}

export interface PerformancePhase {
  name: string;
  durationMs: number;
}

export interface AgentTraceStep {
  id: string;
  timestamp: string;
  toolAction: string;
  status: "authorized" | "completed" | "blocked";
  summary: string;
}

export interface ExpenseRecord {
  id: string;
  tripId: string;
  fileName: string;
  status: ExpenseStatus;
  createdAt: string;
  updatedAt: string;
  extracted: ExtractionResult;
  savedFields: ReceiptFields;
  policyChecks: PolicyCheck[];
  performance: PerformancePhase[];
}

export interface TripRecord {
  id: string;
  employeeName: string;
  tripName: string;
  tripPurpose: string;
  status: TripStatus;
  createdAt: string;
  updatedAt: string;
  totalFiles: number;
  processedFiles: number;
  skippedFiles: Array<{ fileName: string; reason: string }>;
  expenseIds: string[];
  agentTrace: {
    agentRunId: string;
    runtime: "local-agent";
    runtimeControl: "nemoclaw-style-policy";
    steps: AgentTraceStep[];
  };
  approvedAt?: string;
  approvedBy?: string;
}

export interface AppState {
  trips: TripRecord[];
  expenses: ExpenseRecord[];
  auditEvents: AuditEvent[];
}

export interface PolicyConfig {
  companyName: string;
  currency: string;
  requiredFields: string[];
  receiptRequired: boolean;
  businessPurposeRequired: boolean;
  meal: { category: string; attendeesRequiredAbove: number; warnAbove: number; tipWarnPercent: number };
  lodging: { category: string; stayDatesWarning: boolean };
  age: { warnIfOlderThanDays: number };
  cash: { warnIfCashPayment: boolean };
  highAmount: { warnThreshold: number; blockThreshold: number };
}
