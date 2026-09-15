// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import type { FieldProvenance, FieldReasoning, ReceiptFields } from "./types.ts";
import { hasText, parseMoney, titleCase } from "./util.ts";

export interface StructuredReceipt {
  fields: ReceiptFields;
  confidence: Record<string, number>;
  provenance: FieldProvenance[];
  reasoning: FieldReasoning[];
}

export function repairReceiptFromEvidence(structured: StructuredReceipt, evidence: string, fileName = "receipt"): { structured: StructuredReceipt; repairedFields: string[] } {
  const result: StructuredReceipt = {
    fields: { ...structured.fields },
    confidence: { ...structured.confidence },
    provenance: [...structured.provenance],
    reasoning: [...structured.reasoning],
  };
  const repaired = new Set<string>();
  const source = normalizeEvidence(`${fileName}\n${evidence}`);

  const merchant = findMerchant(source, fileName);
  if (merchant && missing(result.fields.merchant)) fill(result, repaired, "merchant", merchant.value, 0.91, merchant.evidence, "Merchant was inferred from the receipt header, merchant line, or filename context.", merchant.rawValue);

  const category = findCategory(source, fileName, result.fields.merchant);
  if (category && (missing(result.fields.category) || result.fields.category === "Other")) fill(result, repaired, "category", category.value, 0.88, category.evidence, "Category was inferred from merchant and receipt context.", category.rawValue, true);

  const amount = findTotalAmount(source);
  if (amount && (missing(result.fields.amount) || shouldReplaceAmount(result.fields.amount, amount, source))) fill(result, repaired, "amount", amount.value, 0.9, amount.evidence, amount.summary ?? "Amount was selected from a final payable total while excluding taxes, line items, balances, deposits, and authorization holds.", amount.rawValue, true);

  const date = findTransactionDate(source);
  if (date && (missing(result.fields.transactionDate) || shouldReplaceDate(result.fields.transactionDate, date))) fill(result, repaired, "transactionDate", date.value, 0.86, date.evidence, "Transaction date was selected from purchase, rental-start, check-closed, or folio evidence.", date.rawValue, true);

  if (missing(result.fields.currency) && /\$/.test(source)) fill(result, repaired, "currency", "USD", 0.82, "Dollar amounts appear on the receipt.", "Currency inferred from dollar amounts.", "$");

  const tax = findTax(source);
  if (tax && missing(result.fields.tax)) fill(result, repaired, "tax", tax.value, 0.84, tax.evidence, "Tax was extracted from an explicit tax line.", tax.rawValue);

  const tip = findTip(source);
  if (tip && missing(result.fields.tip)) fill(result, repaired, "tip", tip.value, 0.84, tip.evidence, "Tip was extracted from an explicit tip or gratuity line.", tip.rawValue);
  if (!tip && String(result.fields.category || "").toLowerCase() === "meals" && missing(result.fields.tip)) fill(result, repaired, "tip", 0, 0.8, "No tip or gratuity line found.", "Tip defaulted to zero because no explicit tip/gratuity was present.", "not present");

  const payment = findPaymentMethod(source);
  if (payment && missing(result.fields.paymentMethod)) fill(result, repaired, "paymentMethod", payment.value, 0.86, payment.evidence, "Payment method was recovered from card, cash, or payment text.", payment.rawValue);

  const location = findLocation(source);
  if (location && missing(result.fields.location)) fill(result, repaired, "location", location.value, 0.82, location.evidence, "Location was recovered from merchant, rental, store, or hotel location evidence.", location.rawValue);

  if (String(result.fields.category || "").toLowerCase() === "lodging") {
    const stay = findStayDates(source);
    if (stay.checkIn && missing(result.fields.checkInDate)) fill(result, repaired, "checkInDate", stay.checkIn.value, 0.86, stay.checkIn.evidence, "Hotel check-in date was recovered from arrival evidence.", stay.checkIn.rawValue);
    if (stay.checkOut && missing(result.fields.checkOutDate)) fill(result, repaired, "checkOutDate", stay.checkOut.value, 0.86, stay.checkOut.evidence, "Hotel check-out date was recovered from departure evidence.", stay.checkOut.rawValue);
  }

  return { structured: result, repairedFields: [...repaired] };
}

type Candidate<T> = { value: T; rawValue: string; evidence: string; score: number; summary?: string };

function fill(result: StructuredReceipt, repaired: Set<string>, field: keyof ReceiptFields, value: string | number, confidence: number, evidence: string, summary: string, rawValue?: string, force = false): void {
  if (!force && !missing(result.fields[field])) return;
  if (force) delete result.confidence[field];
  // Drop any stale entries for this field first so a fill never leaves duplicate provenance/reasoning records.
  result.provenance = result.provenance.filter((item) => item.field !== field);
  result.reasoning = result.reasoning.filter((item) => item.field !== field);
  result.fields[field] = value as never;
  result.confidence[field] = force ? confidence : Math.max(result.confidence[field] ?? 0, confidence);
  result.provenance.push({ field, source: "schema-guard", evidence });
  result.reasoning.push({ field, summary, evidence: [evidence], rawValue: rawValue ?? evidence, normalizedValue: String(value), confidence });
  repaired.add(field);
}

function missing(value: unknown): boolean {
  return value === null || value === undefined || (typeof value === "string" && value.trim() === "");
}

function normalizeEvidence(value: string): string {
  return value.replace(/\r/g, "").replace(/\\&/g, "&").replace(/[\t]+/g, " ").replace(/ {2,}/g, " ");
}

function lines(evidence: string): string[] {
  return evidence.split("\n").map((line) => line.trim().replace(/ {2,}/g, " ")).filter(Boolean);
}

function findMerchant(evidence: string, fileName: string): Candidate<string> | null {
  const merchantLine = lines(evidence).find((line) => /^(merchant|vendor|business)\s*:/i.test(line));
  if (merchantLine) {
    const raw = merchantLine.replace(/^(merchant|vendor|business)\s*:\s*/i, "").trim();
    return { value: titleCase(raw), rawValue: raw, evidence: merchantLine, score: 100 };
  }
  const header = lines(evidence).find((line) => /^[A-Z][A-Z0-9 &'.-]{4,60}$/.test(line) && !/^(receipt|invoice|total|date|time)$/i.test(line));
  if (header) return { value: titleCase(header), rawValue: header, evidence: header, score: 85 };
  const fromName = fileName.replace(/\.(png|jpg|jpeg)$/i, "").replace(/^\d+[-_]/, "").replace(/[-_]/g, " ");
  return fromName ? { value: titleCase(fromName), rawValue: fromName, evidence: `filename: ${fileName}`, score: 60 } : null;
}

function findCategory(evidence: string, fileName: string, merchant: unknown): Candidate<string> | null {
  const text = `${fileName}\n${merchant ?? ""}\n${evidence}`.toLowerCase();
  const categories: Array<{ pattern: RegExp; value: string; label: string; score: number }> = [
    { pattern: /\b(parking|parking charges|parking total|masterpark|lot location|airport access fee|terminal|ticket #|vehicle|license|taxi|rideshare|transportation|fuel|train|rental car|car rental|\brental\b)\b/, value: "Transportation", label: "transportation or parking evidence", score: 110 },
    { pattern: /\b(restaurant|cafe|bistro|meal|dinner|lunch|breakfast|gratuity|tip)\b/, value: "Meals", label: "meal or restaurant evidence", score: 95 },
    { pattern: /\b(hotel|folio|room charge|room type|occupancy tax|check[- ]?in|check[- ]?out|lodging)\b/, value: "Lodging", label: "lodging or hotel evidence", score: 90 },
    { pattern: /\b(arrive|arrival|depart|departure)\b.*\b(room|hotel|folio|guest)\b|\b(room|hotel|folio|guest)\b.*\b(arrive|arrival|depart|departure)\b/, value: "Lodging", label: "lodging stay evidence", score: 90 },
    { pattern: /\b(airline|flight|airfare)\b/, value: "Airfare", label: "airfare evidence", score: 90 },
    { pattern: /\b(supply|office|printer|paper)\b/, value: "Supplies", label: "supply evidence", score: 90 },
  ];
  const match = categories.filter((category) => category.pattern.test(text)).sort((a, b) => b.score - a.score)[0];
  if (match) return { value: match.value, rawValue: match.label, evidence: match.label, score: match.score };
  return { value: "Other", rawValue: "no strong category cue", evidence: "No strong category cue found.", score: 40 };
}

function findTotalAmount(evidence: string): Candidate<number> | null {
  const candidates: Candidate<number>[] = [];
  const evidenceLines = lines(evidence);
  for (const [index, line] of evidenceLines.entries()) {
    const windows = [line, [line, evidenceLines[index + 1]].filter(Boolean).join(" ")];
    for (const text of windows) for (const match of moneyMatches(text)) {
      const candidate = amountCandidate(text, match);
      if (candidate) candidates.push(candidate);
    }
  }
  const folioSettlement = findFolioCardSettlementAmount(evidence, evidenceLines);
  if (folioSettlement) candidates.push(folioSettlement);
  return candidates.sort((a, b) => b.score - a.score || b.value - a.value)[0] ?? null;
}

function findFolioCardSettlementAmount(evidence: string, evidenceLines = lines(evidence)): Candidate<number> | null {
  const source = evidence.toLowerCase();
  const hasLodgingFolioContext = /\b(hotel|folio|room charge|room type|occupancy tax|arrive|arrival|depart|departure|lodging|marriott|residence inn)\b/.test(source);
  if (!hasLodgingFolioContext) return null;

  const candidates: Candidate<number>[] = [];
  for (const line of evidenceLines) {
    const brand = cardBrand(line);
    if (!brand) continue;

    const lower = line.toLowerCase();
    if (/authorization|auth|hold|deposit|balance due|balance:|card #|card type|card entry|approval|aid:|label:/.test(lower)) continue;

    const match = moneyMatches(line).at(-1);
    const value = match ? parseMoney(match[0]) : null;
    if (value === null || value <= 0) continue;

    candidates.push({
      value,
      rawValue: match![0],
      evidence: line.slice(0, 180),
      score: hasZeroBalance(evidence) ? 118 : 108,
      summary: "Amount was inferred from a hotel folio card-settlement row while ignoring balance, tax, fee, room-charge, deposit, and authorization lines.",
    });
  }

  return candidates.sort((a, b) => b.score - a.score || b.value - a.value)[0] ?? null;
}

function hasZeroBalance(evidence: string): boolean {
  return /\bbalance(?:\s+due)?\s*[:&]?\s*(?:\$?\s*)0(?:\.00)?\b/i.test(evidence);
}

function amountCandidate(line: string, match: RegExpMatchArray): Candidate<number> | null {
  const value = parseMoney(match[0]);
  if (value === null || value <= 0) return null;
  const lower = line.toLowerCase();
  const before = line.slice(0, match.index ?? 0).toLowerCase();
  if (/authorization|auth amount|deposit|hold/.test(lower)) return null;
  if (/balance|change due/.test(lower)) return null;
  if (/tax/.test(before) && !/plus tax/.test(lower)) return null;
  if (/tip|gratuity/.test(before) && !/total/.test(before)) return null;
  if (/subtotal/.test(before) && !/grand total|total due/.test(lower)) return null;
  let score = 0;
  if (/grand total|total due|amount due|total estimated charge|total charges|total paid|total - plus tax|final total/.test(lower)) score += 120;
  if (/payment|paid|credit card sale/.test(lower)) score += 85;
  if (/^\s*total\b/.test(lower) || /\btotal\s*[:$]/.test(lower)) score += 90;
  if (/amount\s*:/.test(lower)) score += 85;
  if (/line item|item count|qty/.test(lower)) score -= 80;
  return score > 0 ? { value, rawValue: match[0], evidence: line.slice(0, 180), score } : null;
}

function moneyMatches(value: string): RegExpMatchArray[] {
  return [...value.matchAll(/(?:USD|US\$|\$)\s*[0-9]{1,6}(?:,[0-9]{3})*(?:\.[0-9]{2})?/gi), ...value.matchAll(/\b[0-9]{1,6}(?:,[0-9]{3})*\.[0-9]{2}\b/g)];
}

function shouldReplaceAmount(current: unknown, candidate: Candidate<number>, evidence: string): boolean {
  if (typeof current !== "number" || !Number.isFinite(current)) return true;
  if (Math.abs(current - candidate.value) < 0.01) return false;
  const currentLine = lines(evidence).find((line) => line.includes(current.toFixed(2)) || line.includes(String(current)));
  return Boolean(currentLine && /tax|authorization|hold|balance/.test(currentLine.toLowerCase()));
}

function findTax(evidence: string): Candidate<number> | null {
  return findLabeledMoney(evidence, /\b(total tax|sales tax|tax)\b/i, 70);
}

function findTip(evidence: string): Candidate<number> | null {
  return findLabeledMoney(evidence, /\b(tip|gratuity)\b/i, 70);
}

function findLabeledMoney(evidence: string, label: RegExp, score: number): Candidate<number> | null {
  for (const line of lines(evidence)) {
    if (!label.test(line)) continue;
    const match = moneyMatches(line).at(-1);
    const value = match ? parseMoney(match[0]) : null;
    if (value !== null) return { value, rawValue: match![0], evidence: line.slice(0, 180), score };
  }
  return null;
}

function findTransactionDate(evidence: string): Candidate<string> | null {
  const candidates: Candidate<string>[] = [];
  for (const [index, line] of lines(evidence).entries()) collectDateCandidates(line, index, candidates);
  return candidates.sort((a, b) => b.score - a.score)[0] ?? null;
}

function collectDateCandidates(line: string, lineIndex: number, candidates: Candidate<string>[]): void {
  const patterns: Array<{ regex: RegExp; parse: (match: RegExpMatchArray) => string | null }> = [
    { regex: /\b(\d{1,2})\/(\d{1,2})\/(\d{4}|\d{2})\b/g, parse: (m) => parseDateParts(m[3], m[1], m[2]) },
    { regex: /\b(\d{1,2})[-\s]([A-Za-z]{3,9})[-,\s](\d{4}|\d{2})\b/g, parse: (m) => parseDateParts(m[3], monthNumber(m[2]), m[1]) },
    { regex: /\b(\d{1,2})(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)(\d{4}|\d{2})\b/gi, parse: (m) => parseDateParts(m[3], monthNumber(m[2]), m[1]) },
    { regex: /\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4}|\d{2})\b/gi, parse: (m) => parseDateParts(m[3], monthNumber(m[1]), m[2]) },
  ];
  for (const { regex, parse } of patterns) for (const match of line.matchAll(regex)) {
    const value = parse(match);
    if (value) candidates.push({ value, rawValue: match[0], evidence: line.slice(0, 180), score: dateScore(line, value, lineIndex) });
  }
}

function dateScore(line: string, isoDate: string, lineIndex: number): number {
  const lower = line.toLowerCase();
  let score = 40 - Math.min(lineIndex, 20);
  if (/purchase|transaction|date|check closed|payment|rental start|rental time|arrival/.test(lower)) score += 60;
  if (/depart|checkout|return time/.test(lower)) score -= 25;
  if (isSuspiciousFutureDate(isoDate)) score -= 200;
  return score;
}

function shouldReplaceDate(current: unknown, candidate: Candidate<string>): boolean {
  return typeof current !== "string" || !current || (isSuspiciousFutureDate(current) && !isSuspiciousFutureDate(candidate.value));
}

function parseDateParts(rawYear: string, rawMonth: string | number | null, rawDay: string | number): string | null {
  const month = Number(rawMonth);
  const day = Number(rawDay);
  if (!Number.isInteger(month) || !Number.isInteger(day) || month < 1 || month > 12 || day < 1 || day > 31) return null;
  const year = rawYear.length === 2 ? `20${rawYear}` : rawYear;
  return `${year}-${String(month).padStart(2, "0")}-${String(day).padStart(2, "0")}`;
}

function monthNumber(value: string): number | null {
  const months: Record<string, number> = { jan: 1, feb: 2, mar: 3, apr: 4, may: 5, jun: 6, jul: 7, aug: 8, sep: 9, oct: 10, nov: 11, dec: 12 };
  return months[value.toLowerCase().slice(0, 3)] ?? null;
}

function isSuspiciousFutureDate(value: string): boolean {
  return Number(value.slice(0, 4)) > new Date().getFullYear() + 1;
}

function findPaymentMethod(evidence: string): Candidate<string> | null {
  for (const line of lines(evidence)) {
    if (/cash/i.test(line) && /paid|payment|tender/i.test(line)) return { value: "Cash", rawValue: "cash", evidence: line, score: 90 };
    if (/paid\s*-\s*card|card\s*#|masked card/i.test(line)) return { value: cardBrand(line) ?? "Card", rawValue: line, evidence: line, score: 92 };
    if (/\bcard\b/i.test(line) && /paid|payment|approved|tender/i.test(line)) return { value: cardBrand(line) ?? "Card", rawValue: line, evidence: line, score: 89 };
    const brand = cardBrand(line);
    if (brand && /paid|payment|card|sale|visa|mastercard|amex|discover/i.test(line)) return { value: brand, rawValue: brand, evidence: line, score: 88 };
  }
  return null;
}

function cardBrand(value: string): string | null {
  if (/visa/i.test(value)) return "Visa";
  if (/master\s*card|mastercard/i.test(value)) return "Mastercard";
  if (/american express|amex/i.test(value)) return "American Express";
  if (/discover/i.test(value)) return "Discover";
  return null;
}

function findLocation(evidence: string): Candidate<string> | null {
  const match = evidence.match(/\b(?:location|store|rental location|hotel location)\s*:\s*([^\n]{3,70})/i);
  if (!match) return null;
  const raw = match[1].trim();
  return { value: titleCase(raw), rawValue: raw, evidence: match[0].slice(0, 180), score: 80 };
}

function findStayDates(evidence: string): { checkIn: Candidate<string> | null; checkOut: Candidate<string> | null } {
  const checkIn = lines(evidence).map((line, index) => /arrive|arrival|check-?in/i.test(line) ? firstDate(line, index) : null).find(Boolean) ?? null;
  const checkOut = lines(evidence).map((line, index) => /depart|departure|check-?out/i.test(line) ? firstDate(line, index) : null).find(Boolean) ?? null;
  return { checkIn, checkOut };
}

function firstDate(line: string, index: number): Candidate<string> | null {
  const candidates: Candidate<string>[] = [];
  collectDateCandidates(line, index, candidates);
  return candidates.sort((a, b) => b.score - a.score)[0] ?? null;
}
