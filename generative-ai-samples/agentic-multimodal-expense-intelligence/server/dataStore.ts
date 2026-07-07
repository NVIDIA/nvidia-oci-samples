// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import type { AppState, AuditEvent, ExpenseRecord, TripRecord } from "./types.ts";

export class DataStore {
  rootDir: string;
  stateFile: string;
  uploadsDir: string;

  constructor(rootDir = process.cwd()) {
    this.rootDir = rootDir;
    this.stateFile = join(rootDir, "data", "state.json");
    this.uploadsDir = join(rootDir, "data", "uploads");
  }

  async ensure(): Promise<void> {
    await mkdir(dirname(this.stateFile), { recursive: true });
    await mkdir(this.uploadsDir, { recursive: true });
    await this.readState();
  }

  async readState(): Promise<AppState> {
    try {
      return JSON.parse(await readFile(this.stateFile, "utf8")) as AppState;
    } catch (error) {
      if (isNotFound(error)) {
        const empty: AppState = { trips: [], expenses: [], auditEvents: [] };
        await this.writeState(empty);
        return empty;
      }
      throw error;
    }
  }

  async writeState(state: AppState): Promise<void> {
    await mkdir(dirname(this.stateFile), { recursive: true });
    await writeFile(this.stateFile, `${JSON.stringify(state, null, 2)}\n`);
  }

  async reset(): Promise<void> {
    await this.writeState({ trips: [], expenses: [], auditEvents: [] });
  }

  async mutate(mutator: (state: AppState) => void | Promise<void>): Promise<AppState> {
    const state = await this.readState();
    await mutator(state);
    await this.writeState(state);
    return state;
  }

  async addTrip(trip: TripRecord): Promise<void> {
    await this.mutate((state) => { state.trips.unshift(trip); });
  }

  async updateTrip(trip: TripRecord): Promise<void> {
    await this.mutate((state) => {
      const index = state.trips.findIndex((candidate) => candidate.id === trip.id);
      if (index >= 0) state.trips[index] = trip;
    });
  }

  async addExpense(expense: ExpenseRecord): Promise<void> {
    await this.mutate((state) => { state.expenses.unshift(expense); });
  }

  async updateExpense(expense: ExpenseRecord): Promise<void> {
    await this.mutate((state) => {
      const index = state.expenses.findIndex((candidate) => candidate.id === expense.id);
      if (index >= 0) state.expenses[index] = expense;
    });
  }

  async addAudit(event: AuditEvent): Promise<void> {
    await this.mutate((state) => { state.auditEvents.unshift(event); });
  }

  async trip(id: string): Promise<TripRecord | null> {
    return (await this.readState()).trips.find((trip) => trip.id === id) ?? null;
  }

  async expense(id: string): Promise<ExpenseRecord | null> {
    return (await this.readState()).expenses.find((expense) => expense.id === id) ?? null;
  }

  async tripExpenses(tripId: string): Promise<ExpenseRecord[]> {
    const state = await this.readState();
    return state.expenses.filter((expense) => expense.tripId === tripId).reverse();
  }
}

function isNotFound(error: unknown): boolean {
  return typeof error === "object" && error !== null && "code" in error && (error as { code?: unknown }).code === "ENOENT";
}
