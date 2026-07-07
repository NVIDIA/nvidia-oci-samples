// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import test from "node:test";
import assert from "node:assert/strict";
import { repairReceiptFromEvidence } from "../server/evidenceRepair.ts";

const empty = { fields: {}, confidence: {}, provenance: [], reasoning: [] };

test("selects car rental total instead of authorization hold", () => {
  const evidence = `ABC MOBILITY RENTAL
RENTAL START: 05/11/2026 02:47 PM
RENTAL LOCATION: SAN JOSE AIRPORT
SALES TAX $13.89
TOTAL ESTIMATED CHARGE $183.95
CREDIT CARD AUTHORIZATION AMOUNT $384.00
PAID BY CARDHOLDER/VISA`;
  const { structured } = repairReceiptFromEvidence(empty, evidence, "01-car-rental-authorization-hold.png");
  assert.equal(structured.fields.amount, 183.95);
  assert.equal(structured.fields.transactionDate, "2026-05-11");
  assert.equal(structured.fields.category, "Transportation");
  assert.equal(structured.fields.paymentMethod, "Visa");
});

test("handles Total - Plus Tax and defaults meal tip to zero", () => {
  const evidence = `GARDEN CAFE
DATE: 05/13/2026
LOCATION: ABC CAMPUS CAFE
TOTAL - PLUS TAX $9.75
PAID - CARD# ********7113
BALANCE $0.00`;
  const { structured } = repairReceiptFromEvidence(empty, evidence, "02-cafe-total-plus-tax.png");
  assert.equal(structured.fields.amount, 9.75);
  assert.equal(structured.fields.tip, 0);
  assert.equal(structured.fields.category, "Meals");
  assert.equal(structured.fields.paymentMethod, "Card");
});

test("extracts named month dates and explicit gratuity", () => {
  const evidence = `HARBOR BISTRO
CHECK CLOSED MAY 12, 2026 08:42 PM
SUBTOTAL $92.00
SALES TAX $8.05
GRATUITY $27.60
TOTAL DUE $127.65
PAYMENT MASTERCARD ****4455`;
  const { structured } = repairReceiptFromEvidence(empty, evidence, "03-business-dinner-tip.png");
  assert.equal(structured.fields.transactionDate, "2026-05-12");
  assert.equal(structured.fields.amount, 127.65);
  assert.equal(structured.fields.tip, 27.6);
  assert.equal(structured.fields.paymentMethod, "Mastercard");
});

test("extracts dashed lodging stay dates", () => {
  const evidence = `CLOUDVIEW HOTEL
ARRIVE 11-MAY-2026
DEPART 13-MAY-2026
TOTAL CHARGES $385.12
VISA CREDIT $385.12`;
  const { structured } = repairReceiptFromEvidence(empty, evidence, "04-hotel-folio-stay-dates.png");
  assert.equal(structured.fields.category, "Lodging");
  assert.equal(structured.fields.checkInDate, "2026-05-11");
  assert.equal(structured.fields.checkOutDate, "2026-05-13");
  assert.equal(structured.fields.amount, 385.12);
});

test("infers hotel folio total from card settlement row", () => {
  const evidence = `Residence Inn San Jose North/Silicon Valley
Arrive: 11May26
Depart: 13May26
Room: 119
Folio Number: 85011
BALANCE: 0.00
DATE DESCRIPTION CHARGES CREDITS
11May26 Room Charge 259.00
11May26 Occupancy Tax 25.90
11May26 Convention and Tourism Tax 10.36
11May26 Calif/Local Tourism Fee 0.51
11May26 SJ HBID Fee 1.00
12May26 Room Charge 259.00
12May26 Occupancy Tax 25.90
12May26 Convention and Tourism Tax 10.36
12May26 Calif/Local Tourism Fee 0.51
12May26 SJ HBID Fee 1.00
13May26 Visa 593.54
Card Type: VISA Card Entry: CHIP Approval: 04576D`;
  const { structured } = repairReceiptFromEvidence(empty, evidence, "04-hotel-folio-settlement.png");
  assert.equal(structured.fields.category, "Lodging");
  assert.equal(structured.fields.checkInDate, "2026-05-11");
  assert.equal(structured.fields.checkOutDate, "2026-05-13");
  assert.equal(structured.fields.amount, 593.54);
  assert.equal(structured.fields.paymentMethod, "Visa");
  assert.match(structured.reasoning.find((item) => item.field === "amount")?.summary ?? "", /folio card-settlement/);
});

test("does not choose total tax as main amount", () => {
  const evidence = `DOWNTOWN SNACKS
DATE: 05/13/2026
TOTAL TAX $0.90
AMOUNT: $9.60
PAID VISA CREDIT`;
  const { structured } = repairReceiptFromEvidence(empty, evidence, "snacks.png");
  assert.equal(structured.fields.amount, 9.60);
});

test("leaves obscured final amount for human review", () => {
  const evidence = `SKY HARBOR PARKING
MERCHANT: SKY HARBOR PARKING
TICKET 116
DATE: 05/13/26 08:10 PM
LOCATION: SJC TERMINAL LOT
PARKING CHARGE $44.00
TAX / SURCHARGE OBSCURED
AMOUNT DUE $--.--
PAYMENT CARD APPROVED
NOTE: FINAL PAYABLE AMOUNT IS COVERED BY STAMP AND CANNOT BE READ`;
  const { structured } = repairReceiptFromEvidence(empty, evidence, "05-obscured-parking-human-review.png");
  assert.equal(structured.fields.merchant, "Sky Harbor Parking");
  assert.equal(structured.fields.transactionDate, "2026-05-13");
  assert.equal(structured.fields.category, "Transportation");
  assert.equal(structured.fields.currency, "USD");
  assert.equal(structured.fields.paymentMethod, "Card");
  assert.equal(structured.fields.amount, undefined);
});

test("classifies parking receipts as transportation despite stay wording", () => {
  const evidence = `MASTERPARK - Lot G
16826 International Blvd
Seatac, WA. 98188
Thank you for choosing MASTERPARK!
Ticket # 10157939
Terminal arrival
Open Date 05/11/26 09:22
Close Date 05/13/26 20:37
Vehicle GRAY KIA/TELLURIDE
License BWN7698
Lot Location 2054X
Net Points Earned This Stay 78
PARKING CHARGES
Parking Total $98.54
GRAND TOTAL $98.55
PAYMENTS
VI_7113 05/13/26 20:37 $98.55-
EMV NAME: CAPITAL ONE VISA
Balance Due $0.00`;
  const { structured } = repairReceiptFromEvidence(empty, evidence, "MasterPark_10157939.jpg");
  assert.equal(structured.fields.category, "Transportation");
  assert.equal(structured.fields.amount, 98.55);
  assert.equal(structured.fields.paymentMethod, "Visa");
});

test("forced repairs replace stale confidence and reasoning for overridden fields", () => {
  const structured = {
    fields: { amount: 0.9 },
    confidence: { amount: 0.99 },
    provenance: [{ field: "amount", source: "nemotron-omni" as const, evidence: "TOTAL TAX $0.90" }],
    reasoning: [{ field: "amount", summary: "Amount was tax.", evidence: ["TOTAL TAX $0.90"], rawValue: "$0.90", normalizedValue: "0.9", confidence: 0.99 }],
  };
  const evidence = `DOWNTOWN SNACKS
DATE: 05/13/2026
TOTAL TAX $0.90
AMOUNT: $9.60
PAID VISA CREDIT`;
  const { structured: repaired } = repairReceiptFromEvidence(structured, evidence, "snacks.png");
  assert.equal(repaired.fields.amount, 9.60);
  assert.equal(repaired.confidence.amount, 0.9);
  assert.equal(repaired.reasoning.filter((item) => item.field === "amount").length, 1);
  assert.equal(repaired.provenance.filter((item) => item.field === "amount").length, 1);
});
