// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import type { ExpenseRecord, TripRecord } from "./types.ts";

export function tripCsv(trip: TripRecord, expenses: ExpenseRecord[]): string {
  const headers = ["trip_id", "trip_name", "employee_name", "business_purpose", "receipt_file", "merchant", "date", "amount", "currency", "tax", "tip", "location", "category", "payment_method", "status"];
  const rows = expenses.map((expense) => {
    const f = expense.savedFields;
    return [trip.id, trip.tripName, trip.employeeName, trip.tripPurpose, expense.fileName, f.merchant, f.transactionDate, f.amount, f.currency, f.tax, f.tip, f.location, f.category, f.paymentMethod, expense.status];
  });
  return [headers, ...rows].map((row) => row.map(csvCell).join(",")).join("\n") + "\n";
}

function csvCell(value: unknown): string {
  const text = value === null || value === undefined ? "" : String(value);
  // Neutralize CSV/formula injection: prefix a single quote when a string field
  // begins with a formula trigger so spreadsheet consumers treat it as text (CWE-1236).
  const guarded = typeof value === "string" && /^[=+\-@\t\r]/.test(text) ? `'${text}` : text;
  return /[",\n]/.test(guarded) ? `"${guarded.replaceAll('"', '""')}"` : guarded;
}
