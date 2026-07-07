// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import { mkdir, readdir, unlink, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { deflateSync } from "node:zlib";

const rootDir = join(dirname(fileURLToPath(import.meta.url)), "..");
const manifestDir = join(rootDir, "data", "sample-receipts");
const imageDir = join(manifestDir, "images");

// Single source of truth for generated receipt images and manifest.json.
// Re-run `npm run generate:receipts` after editing this array.
const receipts = [
  {
    fileName: "01-car-rental-authorization-hold.png",
    title: "ABC MOBILITY RENTAL",
    style: "contract",
    parseEvidence: `ABC MOBILITY RENTAL
MERCHANT: ABC MOBILITY RENTAL
RENTAL RECORD # 577946865
RENTAL START: 05/11/2026 02:47 PM
RETURN TIME: 05/13/2026 04:30 PM
RENTAL LOCATION: SAN JOSE AIRPORT
VEHICLE: 2026 ALTIMA
RENTAL RATE 3 @ $40.00 = $120.00
TRANSPORTATION FEE $27.00
SALES TAX $13.89
TOTAL ESTIMATED CHARGE $183.95
CREDIT CARD AUTHORIZATION AMOUNT $384.00
PAID BY CARDHOLDER/VISA`,
  },
  {
    fileName: "02-cafe-total-plus-tax.png",
    title: "GARDEN CAFE",
    style: "thermal",
    parseEvidence: `GARDEN CAFE
MERCHANT: GARDEN CAFE
ORDER # 761
LOCATION: ABC CAMPUS CAFE
DATE: 05/13/2026
TIME: 01:21 PM
DINE IN
1 DOUBLE CHEESEBURGER $4.50
1 CURLY FRIES $2.00
1 GRILLED VEGETABLES $2.00
TOTAL - PLUS TAX $9.75
CUSTOMER NAME: CARDHOLDER/VISA
PAID - CARD# ********7113
BALANCE $0.00`,
  },
  {
    fileName: "03-business-dinner-tip.png",
    title: "HARBOR BISTRO",
    style: "check",
    parseEvidence: `HARBOR BISTRO
MERCHANT: HARBOR BISTRO
CHECK CLOSED MAY 12, 2026 08:42 PM
LOCATION: REDWOOD CITY CA
BUSINESS DINNER
SUBTOTAL $92.00
SALES TAX $8.05
GRATUITY $27.60
TOTAL DUE $127.65
PAYMENT MASTERCARD ****4455`,
  },
  {
    fileName: "04-hotel-folio-stay-dates.png",
    title: "CLOUDVIEW HOTEL",
    style: "folio",
    parseEvidence: `CLOUDVIEW HOTEL
MERCHANT: CLOUDVIEW HOTEL
FOLIO 881104
HOTEL LOCATION: SAN JOSE CA
ARRIVE 11-MAY-2026
DEPART 13-MAY-2026
ROOM CHARGE $340.00
OCCUPANCY TAX $45.12
TOTAL CHARGES $385.12
VISA CREDIT $385.12`,
  },
  {
    fileName: "05-obscured-parking-human-review.png",
    title: "SKY HARBOR PARKING",
    style: "distressed",
    parseEvidence: `SKY HARBOR PARKING
MERCHANT: SKY HARBOR PARKING
TICKET 116
DATE: 05/13/26 08:10 PM
LOCATION: SJC TERMINAL LOT
PARKING CHARGE $44.00
TAX / SURCHARGE OBSCURED
AMOUNT DUE $--.--
PAYMENT CARD APPROVED
NOTE: FINAL PAYABLE AMOUNT IS COVERED BY STAMP AND CANNOT BE READ`,
  },
];

const font = {
  A:["01110","10001","10001","11111","10001","10001","10001"], B:["11110","10001","10001","11110","10001","10001","11110"], C:["01111","10000","10000","10000","10000","10000","01111"], D:["11110","10001","10001","10001","10001","10001","11110"], E:["11111","10000","10000","11110","10000","10000","11111"], F:["11111","10000","10000","11110","10000","10000","10000"], G:["01111","10000","10000","10111","10001","10001","01110"], H:["10001","10001","10001","11111","10001","10001","10001"], I:["11111","00100","00100","00100","00100","00100","11111"], J:["00111","00010","00010","00010","10010","10010","01100"], K:["10001","10010","10100","11000","10100","10010","10001"], L:["10000","10000","10000","10000","10000","10000","11111"], M:["10001","11011","10101","10101","10001","10001","10001"], N:["10001","11001","10101","10011","10001","10001","10001"], O:["01110","10001","10001","10001","10001","10001","01110"], P:["11110","10001","10001","11110","10000","10000","10000"], Q:["01110","10001","10001","10001","10101","10010","01101"], R:["11110","10001","10001","11110","10100","10010","10001"], S:["01111","10000","10000","01110","00001","00001","11110"], T:["11111","00100","00100","00100","00100","00100","00100"], U:["10001","10001","10001","10001","10001","10001","01110"], V:["10001","10001","10001","10001","10001","01010","00100"], W:["10001","10001","10001","10101","10101","10101","01010"], X:["10001","10001","01010","00100","01010","10001","10001"], Y:["10001","10001","01010","00100","00100","00100","00100"], Z:["11111","00001","00010","00100","01000","10000","11111"],
  "0":["01110","10001","10011","10101","11001","10001","01110"], "1":["00100","01100","00100","00100","00100","00100","01110"], "2":["01110","10001","00001","00010","00100","01000","11111"], "3":["11110","00001","00001","01110","00001","00001","11110"], "4":["00010","00110","01010","10010","11111","00010","00010"], "5":["11111","10000","10000","11110","00001","00001","11110"], "6":["01110","10000","10000","11110","10001","10001","01110"], "7":["11111","00001","00010","00100","01000","01000","01000"], "8":["01110","10001","10001","01110","10001","10001","01110"], "9":["01110","10001","10001","01111","00001","00001","01110"],
  " ":["00000","00000","00000","00000","00000","00000","00000"], ".":["00000","00000","00000","00000","00000","01100","01100"], ":":["00000","01100","01100","00000","01100","01100","00000"], "-":["00000","00000","00000","11111","00000","00000","00000"], "/":["00001","00010","00010","00100","01000","01000","10000"], "$": ["00100","01111","10100","01110","00101","11110","00100"], "#":["01010","01010","11111","01010","11111","01010","01010"], "@":["01110","10001","10111","10101","10111","10000","01110"], "*":["00100","10101","01110","11111","01110","10101","00100"], "&":["01100","10010","10100","01000","10101","10010","01101"], "%":["11001","11010","00100","01000","10110","00110","00000"], "(":["00010","00100","01000","01000","01000","00100","00010"], ")":["01000","00100","00010","00010","00010","00100","01000"], "+":["00000","00100","00100","11111","00100","00100","00000"], "'":["00100","00100","01000","00000","00000","00000","00000"], "=":["00000","11111","00000","11111","00000","00000","00000"],
};

await mkdir(imageDir, { recursive: true });
await removeLegacyRootImages(manifestDir);
await removeLegacyRootImages(imageDir);
for (const receipt of receipts) {
  const png = renderReceipt(receipt);
  await writeFile(join(imageDir, receipt.fileName), png);
}
await writeFile(join(manifestDir, "manifest.json"), `${JSON.stringify({
  _spdx: {
    "SPDX-License-Identifier": "Apache-2.0",
    "SPDX-FileCopyrightText": "Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved."
  },
  _generated: "Generated by scripts/generate-sample-receipts.mjs. Edit the receipts array in that script, then rerun npm run generate:receipts.",
  receipts,
}, null, 2)}\n`);
console.log(`Generated ${receipts.length} synthetic receipt images in ${imageDir}`);

async function removeLegacyRootImages(directory) {
  await mkdir(directory, { recursive: true });
  for (const entry of await readdir(directory)) {
    if (/\.png$/i.test(entry)) await unlink(join(directory, entry));
  }
}

function renderReceipt(receipt) {
  const configs = {
    contract: { width: 980, height: 1280, x: 58, y: 64, scale: 3, columns: 50, bg: [248,249,245,255], ink: [22,22,22,255], border: true, header: false },
    thermal: { width: 660, height: 1260, x: 44, y: 58, scale: 3, columns: 32, bg: [252,250,244,255], ink: [35,35,35,255], border: false, header: false },
    check: { width: 820, height: 1060, x: 52, y: 70, scale: 3, columns: 40, bg: [255,254,248,255], ink: [18,24,34,255], border: true, header: true },
    folio: { width: 1260, height: 880, x: 58, y: 62, scale: 3, columns: 72, bg: [250,252,255,255], ink: [18,24,34,255], border: true, header: true },
    ticket: { width: 720, height: 1220, x: 54, y: 72, scale: 4, columns: 28, bg: [246,247,241,255], ink: [16,16,16,255], border: true, header: false },
    distressed: { width: 760, height: 1220, x: 54, y: 70, scale: 4, columns: 30, bg: [235,235,222,255], ink: [58,58,55,255], border: true, header: false },
  };
  const config = configs[receipt.style] ?? configs.contract;
  const { width, height } = config;
  const pixels = Buffer.alloc(width * height * 4, 255);
  drawRect(pixels, width, 0, 0, width, height, [226,230,226,255]);
  drawRect(pixels, width, 26, 24, width - 52, height - 48, config.bg);
  addPaperNoise(pixels, width, height, receipt.fileName);
  if (config.header) {
    drawRect(pixels, width, 26, 24, width - 52, 58, [40,54,68,255]);
    drawText(pixels, width, config.x, 42, receipt.title.toUpperCase(), 3, [255,255,255,255]);
  }
  if (config.border) {
    drawRect(pixels, width, 26, 24, width - 52, 2, [42,42,42,255]);
    drawRect(pixels, width, 26, height - 26, width - 52, 2, [42,42,42,255]);
    drawRect(pixels, width, 26, 24, 2, height - 48, [42,42,42,255]);
    drawRect(pixels, width, width - 28, 24, 2, height - 48, [42,42,42,255]);
  } else {
    for (let x = 35; x < width - 35; x += 22) drawRect(pixels, width, x, 24, 10, 3, [205,205,198,255]);
    for (let x = 35; x < width - 35; x += 22) drawRect(pixels, width, x, height - 28, 10, 3, [205,205,198,255]);
  }
  let y = config.y + (config.header ? 54 : 0);
  for (const rawLine of receipt.parseEvidence.toUpperCase().split("\n")) {
    if (/^(SUBTOTAL|SALES TAX|TOTAL|AMOUNT DUE|VISA|PAYMENT|PAID|CREDIT CARD)/.test(rawLine)) {
      drawDashedLine(pixels, width, config.x, y - 8, Math.min(width - config.x * 2, config.columns * config.scale * 6), config.ink);
      y += 10;
    }
    for (const line of wrap(rawLine, config.columns)) {
      drawText(pixels, width, config.x, y, line, config.scale, config.ink);
      y += config.scale * 11 + (receipt.style === "folio" ? 3 : 0);
    }
    y += receipt.style === "ticket" ? 12 : 8;
  }
  if (receipt.style === "ticket") {
    drawText(pixels, width, width - 280, height - 145, "116", 10, [45,45,45,255]);
  }
  if (receipt.style === "distressed") {
    addHumanReviewDistress(pixels, width, height);
  }
  return png(width, height, pixels);
}

function addHumanReviewDistress(pixels, width, height) {
  drawRotatedStamp(pixels, width, height, "NEEDS REVIEW", [168,38,38,255]);
  drawRect(pixels, width, width - 445, 654, 330, 92, [116,82,52,255]);
  drawRect(pixels, width, width - 416, 676, 264, 40, [54,42,34,255]);
  drawText(pixels, width, 383, 682, "OBSCURED", 3, [236,224,210,255]);
  for (let y = 450; y < 860; y += 9) {
    drawRect(pixels, width, 45, y, width - 90, 1, [205,205,194,255]);
  }
  drawRect(pixels, width, width - 438, 626, 306, 8, [44,44,44,255]);
  drawRect(pixels, width, width - 438, 752, 306, 8, [44,44,44,255]);
  for (let offset = 0; offset < 210; offset += 1) {
    drawRect(pixels, width, 505 + offset, 930 + Math.round(Math.sin(offset / 9) * 12), 7, 5, [38,38,38,255]);
  }
}

function drawRotatedStamp(pixels, width, height, text, color) {
  const scale = 5;
  let cursor = 170;
  const baseY = 520;
  for (const ch of text) {
    const glyph = font[ch] || font[" "];
    for (let row = 0; row < glyph.length; row += 1) {
      for (let col = 0; col < glyph[row].length; col += 1) {
        if (glyph[row][col] !== "1") continue;
        const rawX = cursor + col * scale;
        const rawY = baseY + row * scale;
        const rotatedX = Math.round(rawX + (rawY - baseY) * 0.35);
        const rotatedY = Math.round(rawY - (rawX - 170) * 0.12);
        drawRect(pixels, width, rotatedX, rotatedY, scale, scale, color);
      }
    }
    cursor += 6 * scale;
  }
  drawRect(pixels, width, 118, 472, 520, 7, color);
  drawRect(pixels, width, 118, 572, 520, 7, color);
  drawRect(pixels, width, 118, 472, 7, 107, color);
  drawRect(pixels, width, 631, 472, 7, 107, color);
}

function addPaperNoise(pixels, width, height, seedText) {
  let seed = [...seedText].reduce((sum, ch) => sum + ch.charCodeAt(0), 0);
  for (let i = 0; i < 1800; i += 1) {
    seed = (seed * 1664525 + 1013904223) >>> 0;
    const x = 28 + (seed % Math.max(1, width - 56));
    seed = (seed * 1664525 + 1013904223) >>> 0;
    const y = 26 + (seed % Math.max(1, height - 52));
    const shade = 238 + (seed % 12);
    drawRect(pixels, width, x, y, 1, 1, [shade, shade, shade, 255]);
  }
}

function drawDashedLine(pixels, width, x, y, length, color) {
  for (let offset = 0; offset < length; offset += 22) drawRect(pixels, width, x + offset, y, 14, 2, color);
}

function wrap(line, max) {
  if (line.length <= max) return [line];
  const words = line.split(" ");
  const out = [];
  let current = "";
  for (const word of words) {
    if ((current + " " + word).trim().length > max) { out.push(current); current = word; }
    else current = (current + " " + word).trim();
  }
  if (current) out.push(current);
  return out;
}

function drawText(pixels, width, x, y, text, scale, color) {
  let cursor = x;
  for (const ch of text) {
    const glyph = font[ch] || font[" "];
    for (let row = 0; row < glyph.length; row += 1) {
      for (let col = 0; col < glyph[row].length; col += 1) {
        if (glyph[row][col] === "1") drawRect(pixels, width, cursor + col * scale, y + row * scale, scale, scale, color);
      }
    }
    cursor += 6 * scale;
  }
}

function drawRect(pixels, width, x, y, w, h, color) {
  for (let yy = y; yy < y + h; yy += 1) for (let xx = x; xx < x + w; xx += 1) {
    const i = (yy * width + xx) * 4;
    pixels[i] = color[0]; pixels[i+1] = color[1]; pixels[i+2] = color[2]; pixels[i+3] = color[3];
  }
}

function png(width, height, rgba) {
  const scanlines = Buffer.alloc((width * 4 + 1) * height);
  for (let y = 0; y < height; y += 1) {
    scanlines[y * (width * 4 + 1)] = 0;
    rgba.copy(scanlines, y * (width * 4 + 1) + 1, y * width * 4, (y + 1) * width * 4);
  }
  return Buffer.concat([
    Buffer.from([0x89,0x50,0x4e,0x47,0x0d,0x0a,0x1a,0x0a]),
    chunk("IHDR", Buffer.concat([u32(width), u32(height), Buffer.from([8,6,0,0,0])])),
    chunk("IDAT", deflateSync(scanlines)),
    chunk("IEND", Buffer.alloc(0)),
  ]);
}

function chunk(type, data) {
  const typeBuffer = Buffer.from(type);
  return Buffer.concat([u32(data.length), typeBuffer, data, u32(crc32(Buffer.concat([typeBuffer, data])))]);
}

function u32(value) {
  const b = Buffer.alloc(4);
  b.writeUInt32BE(value >>> 0);
  return b;
}

function crc32(buf) {
  let crc = 0xffffffff;
  for (const byte of buf) {
    crc ^= byte;
    for (let i = 0; i < 8; i += 1) crc = (crc >>> 1) ^ (0xedb88320 & -(crc & 1));
  }
  return (crc ^ 0xffffffff) >>> 0;
}
