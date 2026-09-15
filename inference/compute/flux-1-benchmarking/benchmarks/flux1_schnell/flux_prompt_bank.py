#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import hashlib
import json


REQUEST_PROMPTS = (
    "A cinematic photograph of a black forest at sunrise with detailed mist and warm light",
    "A modern glass house beside a quiet alpine lake beneath dramatic late afternoon clouds",
    "A red vintage bicycle leaning against a blue brick wall after a summer rainstorm",
    "An astronaut exploring a tropical greenhouse filled with luminous flowers and drifting pollen",
    "A handcrafted ceramic teapot on a walnut table lit by a soft studio window",
    "A bustling night market with colorful lanterns reflected across wet cobblestone streets",
    "A snow leopard resting on a rocky mountain ridge beneath a clear star-filled sky",
    "An architectural rendering of a sustainable library surrounded by trees and public gardens",
    "A macro photograph of morning dew suspended across an intricate spider web in sunlight",
    "A friendly robot preparing fresh bread inside a warm rustic countryside kitchen",
    "A wooden sailboat crossing turquoise water near volcanic cliffs under bright morning light",
    "A fashion portrait with sculptural silver fabric against a minimal charcoal studio background",
    "An ancient observatory above desert dunes illuminated by the rising moon and distant stars",
    "A watercolor illustration of a fox reading beneath an oak tree in early autumn",
    "A high-speed photograph of colorful fruit splashing into crystal-clear water against black",
    "A cozy reading room with tall bookshelves a fireplace and sunlight through arched windows",
    "A futuristic electric train arriving at a green city station during a golden sunset",
    "Aerial photography of winding river channels crossing a vivid green coastal wetland landscape",
    "A detailed product photograph of a mechanical wristwatch on textured dark stone and brass",
    "A small mountain village glowing at twilight beneath snow-covered peaks and scattered clouds",
    "A surreal island floating above the ocean with waterfalls falling through pink evening clouds",
    "A documentary photograph of a potter shaping clay in a bright workshop filled with tools",
    "A coral reef alive with tropical fish and sunbeams passing through clear blue water",
    "A minimalist bedroom with natural linen pale wood and soft shadows from morning sunlight",
    "A classic racing car driving through a forest road covered in colorful autumn leaves",
    "A botanical scientific illustration of medicinal herbs arranged neatly on cream parchment",
    "A dramatic lighthouse standing against enormous ocean waves during a dark coastal storm",
    "A playful golden retriever wearing hiking gear beside a tent in a pine forest",
    "An overhead photograph of handmade pasta fresh herbs tomatoes and flour on a kitchen table",
    "A neon-lit alley in a futuristic city with pedestrians carrying transparent umbrellas at night",
    "A peaceful Japanese garden with a stone bridge koi pond and flowering cherry trees",
    "A cinematic wide view of explorers approaching a massive ice cave glowing deep blue",
)


def request_prompts(batch_size: int) -> list[str]:
    if batch_size < 1 or batch_size > len(REQUEST_PROMPTS):
        raise ValueError(
            f"batch_size must be between 1 and {len(REQUEST_PROMPTS)}, got {batch_size}"
        )
    return list(REQUEST_PROMPTS[:batch_size])


def prompt_digest(prompts: list[str]) -> str:
    payload = json.dumps(prompts, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
