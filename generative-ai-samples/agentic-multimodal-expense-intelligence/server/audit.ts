// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import type { AuditEvent, Severity } from "./types.ts";
import { newId, nowIso } from "./util.ts";
import type { DataStore } from "./dataStore.ts";

export class AuditLogger {
  store: DataStore;

  constructor(store: DataStore) {
    this.store = store;
  }

  async write(input: Omit<AuditEvent, "id" | "timestamp" | "severity"> & { severity?: Severity }): Promise<AuditEvent> {
    const event: AuditEvent = {
      id: newId("audit"),
      timestamp: nowIso(),
      severity: input.severity ?? "info",
      type: input.type,
      actor: input.actor,
      action: input.action,
      tripId: input.tripId,
      expenseId: input.expenseId,
      details: input.details,
    };
    await this.store.addAudit(event);
    return event;
  }
}
