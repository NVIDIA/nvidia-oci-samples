<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
-->

# Agentic Multimodal Expense Intelligence

This sample shows how developers can add NVIDIA Nemotron Parse, Nemotron Omni, and NemoClaw-style runtime controls to an enterprise expense workflow on Oracle Cloud Infrastructure (OCI) or a local developer machine.

It is intentionally small and external-safe:

- No runtime npm dependencies.
- No private receipts, API keys, Oracle credentials, Slack credentials, or NVIDIA-specific travel policy.
- Synthetic ABC Company receipts are generated locally.
- By default, receipt processing requires a build.nvidia.com / NVIDIA API key so failures are explicit. The normal developer demo does not silently fall back to local fixture extraction; deterministic fixtures are available only when `MODEL_EXECUTION_MODE=local` is intentionally selected for offline tests.

The demo is not a replacement expense application. It is a reference pattern for integrating agentic multimodal document intelligence into existing ERP, CRM, finance, procurement, or approval workflows.

## What the Demo Shows

A user imports a folder containing five receipt images. The workflow creates one trip report and one expense line per receipt.

The agentic workflow:

1. Reads each receipt image.
2. Calls Nemotron Parse for OCR and layout evidence when configured.
3. Calls Nemotron Omni to normalize receipt evidence into structured fields.
4. Repairs missing or ambiguous fields from Parse evidence when safe.
5. Applies ABC Company policy checks.
6. Routes exceptions to human review.
7. Preserves scoped workflow memory across receipts: Parse evidence, Omni structured outputs, user corrections, policy results, metrics, and audit history.
8. Records a governance/audit trail for model calls, tool actions, policy decisions, redaction, and approval.
9. Produces a CSV export for downstream enterprise systems.

## Why This Goes Beyond OCR

Traditional OCR can read text. This sample demonstrates reasoning over receipt evidence:

- True reimbursable amount vs. authorization hold.
- Final total vs. tax-only lines.
- Hotel folio card-settlement rows vs. balance, tax, fee, and room-charge rows.
- Purchase date vs. return or checkout dates.
- Meals vs. lodging vs. transportation classification.
- Confidence scores and display-safe field rationale.
- Agent decisions about retry, repair, policy checks, and human review.

## Requirements

- Git.
- Node.js 24 or newer.
- npm, which is included with Node.js.
- A modern browser.
- NVIDIA API key for the default hosted Nemotron demo path.

No runtime npm packages are required.

## Quickstart

Verify local prerequisites:

```bash
git --version
node --version
npm --version
```

If any command is missing, install Git from [git-scm.com/downloads](https://git-scm.com/downloads) and Node.js 24 or newer from [nodejs.org](https://nodejs.org/). npm is included with Node.js. Close and reopen the terminal so PATH updates are applied.

Clone the public sample repository:

```bash
git clone https://github.com/NVIDIA/nvidia-oci-samples.git
```

Run on macOS or Linux:

```bash
cd nvidia-oci-samples/generative-ai-samples/agentic-multimodal-expense-intelligence
npm run generate:receipts
npm test
npm run dev
```

Run on Windows PowerShell:

```powershell
cd .\nvidia-oci-samples\generative-ai-samples\agentic-multimodal-expense-intelligence
npm run generate:receipts
npm test
npm run dev
```

Open [http://127.0.0.1:8790](http://127.0.0.1:8790).

Use the browser folder picker to select the image-only fixture folder:

```text
data/sample-receipts/images/
```

The app will process the five generated PNG receipts and show progress after each file.

For the normal demo flow, first open the **Help** tab and follow the build.nvidia.com API-key setup steps. Then open the **API Keys** tab, paste one hosted NVIDIA key, save it for the browser session, and import the folder. The same key is used for Nemotron Parse and Nemotron Omni calls. The key is sent only with receipt-processing requests and is not written to demo state or audit logs.

## Getting A build.nvidia.com API Key

1. Open [build.nvidia.com](https://build.nvidia.com/) and sign in. If you don't have an account, create a new one.
2. Use the Models area to open a Nemotron Parse or Nemotron Omni model page.
3. Go to the [build.nvidia.com/models](https://build.nvidia.com/models) tab, search for "nemotron parse" and create an API key by clicking the "Get API Key" button. Copy the generated `nvapi-...` value locally; do not commit it to source control.
4. Paste the generated `nvapi-...` value into the app's **API Keys** tab.
5. Do not commit the key to source control. For workshops, browser-session keys are the preferred path because each developer can use their own key.

## Why build.nvidia.com Matters

build.nvidia.com is useful beyond this expense demo:

- It exposes cloud-hosted NVIDIA NIM endpoints for prototyping.
- The public model catalog includes language and reasoning models, multimodal/image-to-text models, OCR/document models, retrieval/RAG, code generation, speech, visual generation, safety, healthcare, weather, and physical-AI use cases.
- The site also includes Skills and Blueprints so developers can move from a model call to application patterns and deployable reference workflows.
- This sample uses one pattern: Parse creates receipt evidence, Omni reasons over that evidence, and NemoClaw-style controls govern model routing, tool authorization, audit, redaction, and human approval.

## Optional Nemotron Configuration

Copy `.env.example` to `.env` or export environment variables before starting the app.

```bash
export NVIDIA_API_KEY="nvapi-..."
export MODEL_EXECUTION_MODE=auto
npm run dev
```

The model path is selected as follows:

- `MODEL_EXECUTION_MODE=auto`: default developer path; call Nemotron when browser-session or server keys are present and fail clearly when no keys are available.
- `MODEL_EXECUTION_MODE=local`: explicit offline test mode that uses deterministic fixture evidence and does not call Nemotron.
- `MODEL_EXECUTION_MODE=nemotron`: require Nemotron Parse and Omni calls; fail clearly if keys or endpoints are unavailable.

Keys can come from either:

- the browser **API Keys** tab, which is best for workshops where each developer uses their own build.nvidia.com key; or
- server environment variables, which is best for a shared local demo machine.

When the API Keys tab has a key and `MODEL_EXECUTION_MODE=auto`, each uploaded receipt image is sent through Nemotron Parse and Nemotron Omni. Without a key, the import fails with a clear configuration error. The deterministic fixture path is intentionally available only through `MODEL_EXECUTION_MODE=local` for offline development and tests, not as the default demo behavior.

## Environment Variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `APP_HOST` | `127.0.0.1` | Local HTTP host. |
| `APP_PORT` | `8790` | Local HTTP port. |
| `MODEL_EXECUTION_MODE` | `auto` | `auto`, `local`, or `nemotron`. |
| `NVIDIA_API_KEY` | unset | Shared hosted NVIDIA API key. |
| `NEMOTRON_PARSE_API_KEY` | `NVIDIA_API_KEY` | Optional separate Parse key. |
| `NEMOTRON_OMNI_API_KEY` | `NVIDIA_API_KEY` | Optional separate Omni key. |
| `NEMOTRON_PARSE_BASE_URL` | `https://integrate.api.nvidia.com` | Parse endpoint base URL. |
| `NEMOTRON_OMNI_BASE_URL` | `https://integrate.api.nvidia.com` | Omni endpoint base URL. |
| `NEMOTRON_PARSE_MODEL` | `nvidia/nemotron-parse` | Parse model. |
| `NEMOTRON_OMNI_MODEL` | `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` | Omni model. |

## Demo Flow

For a 15-minute presentation:

1. Show the folder import with five ABC Company receipts.
2. Point out per-file progress and phase timings.
3. Open the car rental receipt to show authorization hold vs. reimbursable amount reasoning.
4. Open the meal receipt to show total vs. tax/tip reasoning.
5. Open the obscured parking receipt to show the agent refusing to guess and routing the line to human review.
6. Edit the missing amount and save it to demonstrate human control.
7. Expand the governance trace to show runtime controls and audit events.
8. Show performance and stats.
9. Download the CSV export as the downstream enterprise handoff.

## Synthetic Receipt Fixtures

The fixture generator creates five public-safe PNG receipts:

- Car rental: final charge vs. authorization hold.
- Cafe meal: `Total - Plus Tax` and card payment.
- Business dinner: explicit tip and attendees policy warning.
- Hotel folio: arrival/departure dates.
- Obscured parking: transportation classification with a covered final amount, forcing human review instead of a guessed payable total.

The image pixels and receipt data are generated from scratch for this sample. The five receipts intentionally use different visual layouts: narrow thermal receipt, landscape hotel folio, distressed parking ticket, restaurant check, and contract-style car rental receipt. Some receipts may have obscured content that is hard for OCR to detect; the distressed parking ticket demonstrates that Parse/Omni should recover visible context, but the workflow should stop before guessing a final amount and route the receipt to human review.

## Tests

```bash
npm test
```

The tests cover:

- Evidence repair for hard receipt cases.
- ABC Company policy decisions.
- Trip workflow grouping and CSV export.
- Nemotron client response parsing helpers.

## Contributing This Sample

Follow the repository contribution flow:

```bash
git checkout main
git pull origin main
git checkout -b feature/agentic-multimodal-expense-intelligence
git add generative-ai-samples/agentic-multimodal-expense-intelligence README.md
git commit -s -m "Add agentic multimodal expense intelligence sample"
git push origin feature/agentic-multimodal-expense-intelligence
```

Open a pull request to `NVIDIA/nvidia-oci-samples:main` and tag the maintainers listed in the root `MAINTAINERS.md`.
