// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import test from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, readFile, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { DataStore } from "../server/dataStore.ts";
import type { ExpenseRecord, TripRecord } from "../server/types.ts";

test("missing state file initializes an empty state", async () => {
  const root = await mkdtemp(join(tmpdir(), "expense-store-missing-"));
  const store = new DataStore(root);
  await store.ensure();
  const state = await store.readState();
  assert.deepEqual(state, { trips: [], expenses: [], auditEvents: [] });
});

test("corrupt state file surfaces the parse failure instead of resetting data", async () => {
  const root = await mkdtemp(join(tmpdir(), "expense-store-corrupt-"));
  const store = new DataStore(root);
  await store.ensure();
  await writeFile(store.stateFile, "{ not json");
  await assert.rejects(() => store.readState(), SyntaxError);
  assert.equal(await readFile(store.stateFile, "utf8"), "{ not json");
});

test("concurrent mutations all persist without lost updates", async () => {
  const root = await mkdtemp(join(tmpdir(), "expense-store-concurrent-"));
  const store = new DataStore(root);
  await store.ensure();
  const count = 25;
  await Promise.all([
    ...Array.from({ length: count }, (_, index) => store.addTrip({ id: `trip_${index}` } as TripRecord)),
    ...Array.from({ length: count }, (_, index) => store.addExpense({ id: `exp_${index}`, tripId: "trip_0" } as ExpenseRecord)),
  ]);
  const state = await store.readState();
  assert.equal(state.trips.length, count);
  assert.equal(state.expenses.length, count);
  assert.equal(new Set(state.trips.map((trip) => trip.id)).size, count);
  assert.equal(new Set(state.expenses.map((expense) => expense.id)).size, count);
});
