// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import type { ExpenseRecord, PolicyCheck, PolicyConfig, ReceiptFields } from "./types.ts";
import { hasText, parseMoney } from "./util.ts";

export function evaluatePolicy(expense: Pick<ExpenseRecord, "savedFields">, tripPurpose: string, policy: PolicyConfig, now = new Date()): PolicyCheck[] {
  const fields = expense.savedFields;
  const checks: PolicyCheck[] = [];
  if (policy.receiptRequired) checks.push({ id: "receipt-required", decision: fields.receiptFileRef ? "pass" : "block", reason: fields.receiptFileRef ? "Receipt file reference is present." : "Receipt file is required.", evidence: { receiptFileRef: fields.receiptFileRef } });
  for (const field of policy.requiredFields) checks.push(requiredFieldCheck(fields, field));
  if (policy.businessPurposeRequired) checks.push({ id: "business-purpose-required", decision: hasText(tripPurpose) ? "pass" : "block", reason: hasText(tripPurpose) ? "Business purpose is present." : "Business purpose is required.", evidence: { tripPurpose } });
  const amount = parseMoney(fields.amount);
  const category = String(fields.category || "Other");
  if (amount !== null && category === policy.meal.category) {
    checks.push({ id: "meal-attendees-threshold", decision: amount > policy.meal.attendeesRequiredAbove ? "warn" : "pass", reason: amount > policy.meal.attendeesRequiredAbove ? `ABC Company asks for attendees when meals exceed USD ${policy.meal.attendeesRequiredAbove}.` : "Meal is below attendee-warning threshold.", evidence: { amount, threshold: policy.meal.attendeesRequiredAbove } });
    checks.push({ id: "meal-manager-review", decision: amount > policy.meal.warnAbove ? "warn" : "pass", reason: amount > policy.meal.warnAbove ? "Meal exceeds manager-review guidance." : "Meal is within manager-review guidance.", evidence: { amount, threshold: policy.meal.warnAbove } });
  }
  if (amount !== null && typeof fields.tip === "number" && amount > 0 && category === policy.meal.category) {
    const percent = Math.round((fields.tip / Math.max(0.01, amount - fields.tip)) * 1000) / 10;
    checks.push({ id: "tip-percent", decision: percent > policy.meal.tipWarnPercent ? "warn" : "pass", reason: percent > policy.meal.tipWarnPercent ? `Tip is above ${policy.meal.tipWarnPercent}% guidance.` : "Tip is within guidance or absent.", evidence: { tip: fields.tip, percent } });
  }
  if (category === policy.lodging.category && policy.lodging.stayDatesWarning) checks.push({ id: "lodging-stay-dates", decision: fields.checkInDate && fields.checkOutDate ? "pass" : "warn", reason: fields.checkInDate && fields.checkOutDate ? "Lodging stay dates are present." : "Lodging stay dates are useful for downstream review.", evidence: { checkInDate: fields.checkInDate, checkOutDate: fields.checkOutDate } });
  if (policy.cash.warnIfCashPayment && String(fields.paymentMethod || "").toLowerCase() === "cash") checks.push({ id: "cash-payment", decision: "warn", reason: "Cash reimbursement may require extra review.", evidence: { paymentMethod: fields.paymentMethod } });
  if (amount !== null) checks.push({ id: "high-amount", decision: amount >= policy.highAmount.blockThreshold ? "block" : amount >= policy.highAmount.warnThreshold ? "warn" : "pass", reason: amount >= policy.highAmount.blockThreshold ? "Amount exceeds hard-stop threshold." : amount >= policy.highAmount.warnThreshold ? "Amount exceeds review threshold." : "Amount is below high-value thresholds.", evidence: { amount, warnThreshold: policy.highAmount.warnThreshold, blockThreshold: policy.highAmount.blockThreshold } });
  if (fields.transactionDate) {
    const parsedDate = new Date(`${fields.transactionDate}T00:00:00Z`);
    const validDate = /^\d{4}-\d{2}-\d{2}$/.test(fields.transactionDate) && Number.isFinite(parsedDate.getTime());
    if (!validDate) {
      checks.push({ id: "expense-age", decision: "warn", reason: "Transaction date could not be parsed.", evidence: { transactionDate: fields.transactionDate } });
    } else {
      const ageDays = Math.floor((now.getTime() - parsedDate.getTime()) / 86_400_000);
      checks.push({ id: "expense-age", decision: ageDays > policy.age.warnIfOlderThanDays ? "warn" : "pass", reason: ageDays > policy.age.warnIfOlderThanDays ? "Expense is older than ABC Company submission guidance." : "Expense date is within submission guidance.", evidence: { ageDays, warnIfOlderThanDays: policy.age.warnIfOlderThanDays } });
    }
  }
  return checks;
}

export function statusFromChecks(checks: PolicyCheck[]): "ready_for_review" | "blocked" {
  return checks.some((check) => check.decision === "block") ? "blocked" : "ready_for_review";
}

function requiredFieldCheck(fields: ReceiptFields, field: string): PolicyCheck {
  const value = (fields as Record<string, unknown>)[field];
  const present = value !== null && value !== undefined && value !== "";
  return { id: `required-${field}`, decision: present ? "pass" : "block", reason: present ? `${field} is present.` : `${field} is required for downstream expense staging.`, evidence: { value } };
}
