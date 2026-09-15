// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import test from "node:test";
import assert from "node:assert/strict";
import { Readable } from "node:stream";
import type { IncomingMessage } from "node:http";
import { readRequestJson } from "../server/util.ts";

test("readRequestJson rejects oversized request bodies", async () => {
  const oversized = Readable.from([Buffer.alloc((25 * 1024 * 1024) + 1)]) as IncomingMessage;
  await assert.rejects(() => readRequestJson(oversized), /Request body too large/);
});
