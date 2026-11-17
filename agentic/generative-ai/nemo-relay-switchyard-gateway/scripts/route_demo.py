#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Send the same prompts through each Switchyard route on the Relay gateway.

Routes are selected by requesting the route id as the model name. Routed requests do not
need a client credential: the Switchyard plugin holds the OCI Generative AI API key. The
final direct call to a model id is forwarded to OCI as-is and needs the client's own key,
so it runs only when OCI_GENAI_API_KEY is set.
"""
import argparse
import os
import sys
import time

from openai import OpenAI

EASY = "Rewrite this sentence in title case and nothing else: the quick brown fox jumps over the lazy dog"
MEDIUM = "Explain in three sentences why TCP uses a three-way handshake."
HARD = (
    "Design a lock-free multi-producer multi-consumer bounded queue in C11. Give the full data "
    "structure, enqueue and dequeue with explicit memory orderings, and argue correctness under "
    "the C11 memory model including the ABA problem."
)
PROMPTS = {"easy": EASY, "medium": MEDIUM, "hard": HARD}

CASES = [
    ("oci/efficient", "easy"),
    ("oci/capable", "easy"),
    ("oci/auto", "medium"),
    ("oci/task", "easy"),
    ("oci/task", "medium"),
    ("oci/task", "hard"),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:4040/v1")
    parser.add_argument("--max-tokens", type=int, default=600)
    parser.add_argument("--direct-model", default="openai.gpt-oss-20b",
                        help="model id for the direct (non-routed) call; '' to skip")
    args = parser.parse_args()

    key = os.environ.get("OCI_GENAI_API_KEY")
    routed = OpenAI(base_url=args.base_url, api_key="relay-routed")
    print(f"{'route':14s} {'prompt':7s} {'served model':42s} {'in':>6s} {'out':>6s} {'ms':>7s}  answer")
    failures = 0
    for route, tag in CASES:
        started = time.time()
        try:
            r = routed.chat.completions.create(
                model=route,
                messages=[{"role": "user", "content": PROMPTS[tag]}],
                max_tokens=args.max_tokens,
            )
            text = (r.choices[0].message.content or "").replace("\n", " ")[:60]
            print(f"{route:14s} {tag:7s} {r.model:42s} {r.usage.prompt_tokens:6d} {r.usage.completion_tokens:6d} "
                  f"{int((time.time() - started) * 1000):7d}  {text}")
        except Exception as exc:  # noqa: BLE001 - report and continue
            failures += 1
            print(f"{route:14s} {tag:7s} FAILED: {str(exc)[:160]}")

    if args.direct_model and key:
        started = time.time()
        direct = OpenAI(base_url=args.base_url, api_key=key)
        r = direct.chat.completions.create(
            model=args.direct_model,
            messages=[{"role": "user", "content": PROMPTS["easy"]}],
            max_tokens=args.max_tokens,
        )
        text = (r.choices[0].message.content or "").replace("\n", " ")[:60]
        print(f"{'(direct)':14s} {'easy':7s} {r.model:42s} {r.usage.prompt_tokens:6d} {r.usage.completion_tokens:6d} "
              f"{int((time.time() - started) * 1000):7d}  {text}")
    elif args.direct_model:
        print("(direct)       skipped: set OCI_GENAI_API_KEY to also exercise a direct model call")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
