# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Report exactly which layers were quantized, and check the exclusion filter.

The question this answers is "did the public Model Optimizer recipe do what it
claims", which matters because it is the first question anyone adopting this
will ask before
adopting it.

Three things are reported, in order of how much they can be trusted:

1. **Which modules carry 4-bit weights and which stayed at high precision.**
   Read straight off the tensor dtypes, so it is fact, not inference. Packed
   NVFP4 weights are stored as ``U8``; anything still ``BF16`` was not quantized.

2. **Whether the exclusion filter agrees.** If ``MODELOPT_DIFFUSERS_DIR`` points
   at a Model-Optimizer checkout, the real ``filter_func`` is imported and
   evaluated against every module name we found. That is authoritative: it is the
   same code the export ran. Any layer the filter wanted excluded but which
   carries 4-bit weights, or vice versa, is a genuine defect and is reported as
   a disagreement.

3. **A comparison against the published checkpoint**, which is weak evidence and
   labelled as such. Black Forest Labs ship ComfyUI format and we produce
   Diffusers, so tensor names do not line up and the tensor sets cover different
   components. Only the transformer is compared, and only on precision mix.

Why not just pattern-match layer names: leaf names repeat. FLUX has a model-level
``proj_out`` **and** a ``proj_out`` inside all 38 single transformer blocks. A
substring test cannot tell them apart, and reporting the per-block projections as
wrongly-quantized would be a false alarm — they are ordinary large GEMMs and
quantizing them is the point. Module depth is tracked so the two never merge.
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from common import paths
from typing import Any, Callable

# Layer names Model Optimizer's FLUX filter keeps at high precision. Used only
# for reporting when the real filter is unavailable; item 2 above supersedes it.
DOCUMENTED_EXCLUSIONS = (
    "proj_out",
    "time_text_embed",
    "context_embedder",
    "x_embedder",
    "norm_out",
    "time_guidance_embed",
    "stream_modulation",
)

# Packed sub-8-bit weights land in an unsigned byte tensor. NVFP4 puts two 4-bit
# values per byte; INT8 uses one. Either way the weight is no longer BF16.
PACKED_WEIGHT_DTYPES = frozenset({"U8", "I8", "F8_E4M3", "F8_E5M2"})

# Suffixes that mark scaling metadata rather than a weight.
SCALE_SUFFIXES = (
    "weight_scale",
    "weight_scale_2",
    "input_scale",
    "output_scale",
    "weight_zero_point",
    "_amax",
)


def _read_headers(directory: Path) -> dict[str, dict[str, Any]]:
    """Tensor metadata read straight out of the safetensors header.

    Parses the header bytes rather than calling ``safe_open``. The library
    memory-maps the entire file, which costs address space proportional to the
    checkpoint even though not one weight is ever touched. A 6.8 GB export then
    dies with ``Cannot allocate memory`` on any host with a modest address-space
    limit -- a shared login node, a container with a cgroup cap, a CI runner --
    despite the stage only ever wanting a few kilobytes of JSON.

    The format is fixed and simple: a little-endian ``u64`` header length,
    followed by that many bytes of UTF-8 JSON mapping each tensor name to its
    dtype, shape and byte offsets. Reading it directly costs kilobytes and needs
    neither ``safetensors`` nor ``torch``.

    dtype strings are the safetensors native spellings -- ``F32``, ``BF16``,
    ``U8``, ``F8_E4M3`` -- which is exactly what ``get_dtype()`` returned, so
    nothing downstream changes.
    """
    tensors: dict[str, dict[str, Any]] = {}
    for path in sorted(directory.rglob("*.safetensors")):
        try:
            component = path.relative_to(directory).parts[0]
        except (ValueError, IndexError):
            component = path.parent.name
        if component.endswith(".safetensors"):
            component = "root"

        file_size = path.stat().st_size
        with path.open("rb") as handle:
            length_bytes = handle.read(8)
            if len(length_bytes) < 8:
                continue
            header_length = int.from_bytes(length_bytes, "little")
            # The length is 8 unvalidated bytes from the file. A truncated or
            # corrupt checkpoint can declare a header larger than the file --
            # up to 2**64 -- and the read below would try to allocate it. Bound
            # it by what is actually there before reading a byte.
            if header_length <= 0 or header_length > file_size - 8:
                raise RuntimeError(
                    f"Implausible safetensors header in {path}: declares "
                    f"{header_length} bytes in a {file_size}-byte file."
                )
            try:
                header = json.loads(handle.read(header_length))
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise RuntimeError(f"Unreadable safetensors header in {path}: {exc}") from exc

        for key, spec in header.items():
            # __metadata__ is a free-form string map, not a tensor entry.
            if key == "__metadata__" or not isinstance(spec, dict):
                continue
            tensors[f"{component}/{key}"] = {
                "component": component,
                "name": key,
                "shape": list(spec.get("shape", [])),
                "dtype": str(spec.get("dtype", "")),
                "file": path.name,
            }
    return tensors


def _module_of(tensor_name: str) -> str:
    """Strip the parameter suffix, leaving the module that owns it."""
    for suffix in (*SCALE_SUFFIXES, "weight", "bias"):
        marker = "." + suffix
        if tensor_name.endswith(marker):
            return tensor_name[: -len(marker)]
    return tensor_name.rsplit(".", 1)[0] if "." in tensor_name else tensor_name


def _collapse(module_path: str) -> str:
    """Replace block indices with N so repeated blocks group into one row."""
    return re.sub(r"\.\d+(?=\.|$)", ".N", module_path)


def _depth(module_path: str) -> int:
    return module_path.count(".")


def classify_modules(tensors: dict[str, dict[str, Any]], component: str) -> dict[str, str]:
    """Map each weight-bearing module to 'quantized' or 'high-precision'.

    Read from dtypes alone. A module is quantized when its ``.weight`` is stored
    in a packed low-precision dtype; it is high-precision when the weight is
    still BF16/FP32. Modules with no weight at all (norms without affine terms,
    activations) are absent from the result rather than guessed at.
    """
    states: dict[str, str] = {}
    for record in tensors.values():
        if record["component"] != component:
            continue
        name = record["name"]
        if not name.endswith(".weight"):
            continue
        module = _module_of(name)
        states[module] = (
            "quantized" if record["dtype"] in PACKED_WEIGHT_DTYPES else "high-precision"
        )
    return states


# Files tried first, cheapest first. Everything else in the directory is tried
# afterwards, because upstream moves this function between releases -- 0.42.0
# put it in models_utils.py, which a hardcoded list missed entirely.
# quantize.py is forced last: importing it pulls in the ONNX toolchain at module
# scope, which is slow and may not be installed.
FILTER_SOURCE_PREFERRED = ("utils.py", "models_utils.py", "flux_utils.py", "config.py")
FILTER_SOURCE_LAST = ("quantize.py",)

# Exact names in preference order, then any callable named filter_func*.
FILTER_ATTRIBUTES = ("filter_func_flux_dev", "filter_func_flux", "filter_func")


def _candidate_filter_files(root: Path) -> list[Path]:
    """Every .py in the directory, preferred names first and quantize.py last."""
    everything = sorted(p for p in root.glob("*.py") if p.is_file())
    ordered: list[Path] = []
    for name in FILTER_SOURCE_PREFERRED:
        path = root / name
        if path.is_file():
            ordered.append(path)
    ordered += [
        p
        for p in everything
        if p.name not in FILTER_SOURCE_PREFERRED and p.name not in FILTER_SOURCE_LAST
    ]
    ordered += [root / n for n in FILTER_SOURCE_LAST if (root / n).is_file()]
    return ordered


def _load_modelopt_filter() -> tuple[Callable[[str], bool] | None, str]:
    """Load the real exclusion filter from a Model-Optimizer checkout.

    Returns the callable and where it came from, or ``None`` and the reason it
    could not be loaded. Never raises: an unavailable filter downgrades the
    report, it does not fail the stage.

    Loaded by explicit file path rather than by name. A plain import of
    ``quantize`` would find *this repository's* ``quantize.py`` — it is the
    running entry point and already sits in ``sys.modules`` — and executing our
    own CLI as a side effect of a schema check would be a nasty surprise.
    """
    import importlib.util

    directory = os.environ.get("MODELOPT_DIFFUSERS_DIR")
    if not directory:
        return None, "MODELOPT_DIFFUSERS_DIR not set"
    root = Path(directory)
    if not root.is_dir():
        return None, f"MODELOPT_DIFFUSERS_DIR does not exist: {root}"

    # The example modules import each other by bare name, so the directory has to
    # be importable while we load them.
    inserted = str(root) not in sys.path
    if inserted:
        sys.path.insert(0, str(root))
    # Import failures are recorded rather than swallowed. Silently skipping them
    # made "no filter function found" mean three different things -- the file is
    # absent, the file failed to import, or the name is not in our list -- with
    # nothing to tell them apart.
    failures: list[str] = []

    try:
        for path in _candidate_filter_files(root):
            alias = f"_modelopt_example_{path.stem}"
            try:
                spec = importlib.util.spec_from_file_location(alias, path)
                if spec is None or spec.loader is None:
                    failures.append(f"{path.name}: no import spec")
                    continue
                module = importlib.util.module_from_spec(spec)
                sys.modules[alias] = module
                spec.loader.exec_module(module)
            except Exception as exc:  # noqa: BLE001 - reported below
                sys.modules.pop(alias, None)
                failures.append(f"{path.name}: {type(exc).__name__}: {exc}")
                continue

            for attribute in FILTER_ATTRIBUTES:
                candidate = getattr(module, attribute, None)
                if callable(candidate):
                    return candidate, f"{path.name}:{attribute} in {root}"

            # Nothing under a known name -- accept any callable filter_func*.
            for attribute in sorted(dir(module)):
                if attribute.startswith("filter_func") and callable(getattr(module, attribute)):
                    return getattr(module, attribute), f"{path.name}:{attribute} in {root}"

            sys.modules.pop(alias, None)

        detail = f" ({len(failures)} file(s) failed to import: {'; '.join(failures[:3])})" if failures else ""
        return None, f"no filter function found in {root}{detail}"
    finally:
        if inserted and str(root) in sys.path:
            sys.path.remove(str(root))


def check_filter_agreement(
    states: dict[str, str], filter_func: Callable[[str], bool]
) -> dict[str, Any]:
    """Compare what the filter wanted against what the export actually contains.

    Model Optimizer's convention is that ``filter_func(name)`` returning True
    means *exclude from quantization*. A disagreement in either direction is a
    real defect, so both are reported separately rather than summed.
    """
    excluded_but_quantized: list[str] = []
    included_but_untouched: list[str] = []
    errors: list[str] = []

    for module, state in sorted(states.items()):
        try:
            wants_exclusion = bool(filter_func(module))
        except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
            # A filter that raises tells us nothing about this module, so it must
            # not be counted as agreement. Skipping silently and still reporting
            # ``checked: len(states)`` would let a filter that raises on *every*
            # module report "VERIFIED, 656 modules checked" having checked none.
            errors.append(f"{module}: {type(exc).__name__}: {exc}")
            continue
        if wants_exclusion and state == "quantized":
            excluded_but_quantized.append(module)
        elif not wants_exclusion and state == "high-precision":
            included_but_untouched.append(module)

    evaluated = len(states) - len(errors)

    return {
        "checked": evaluated,
        "total_modules": len(states),
        "errors": errors[:10],
        "error_count": len(errors),
        "excluded_but_quantized": excluded_but_quantized,
        "included_but_high_precision": included_but_untouched,
        # Agreement requires that the filter actually ran on every module. A
        # partial evaluation is an inconclusive result, not a passing one.
        "agrees": not excluded_but_quantized and not errors and evaluated > 0,
    }


def exclusion_report(states: dict[str, str], names: tuple[str, ...]) -> dict[str, Any]:
    """Report documented-exclusion names by where they sit, without judging them.

    Position in the module path is what distinguishes the two cases, so matching
    is done per path segment rather than on the leaf:

    * ``proj_out`` and ``time_text_embed.timestep_embedder.linear_1`` both begin
      with a protected name, so they are the model-level modules the filter is
      written to protect — including everything beneath one.
    * ``single_transformer_blocks.0.proj_out`` only matches at depth 2. It is a
      per-block projection that happens to share a leaf name, and quantizing it
      is intended.

    Merging those two would report 38 healthy GEMMs as a recipe defect.
    """
    protected_subtrees: dict[str, str] = {}
    nested: dict[str, Counter] = defaultdict(Counter)

    for module, state in states.items():
        segments = module.split(".")
        position = next((i for i, seg in enumerate(segments) if seg in names), None)
        if position is None:
            continue
        if position == 0:
            protected_subtrees[module] = state
        else:
            nested[_collapse(module)][state] += 1

    return {
        "top_level": dict(sorted(protected_subtrees.items())),
        "nested": {k: dict(v) for k, v in sorted(nested.items())},
    }


def rank_breakdown(
    tensors: dict[str, dict[str, Any]], component: str, modules: list[str]
) -> Counter:
    """Split a set of modules by the rank of their weight.

    Explains why a layer the filter permits was still left at high precision.
    A 1-D weight is a normalization scale vector — there is no matrix multiply,
    so a low-precision GEMM kernel has nothing to accelerate and Model Optimizer
    does not convert it. FLUX carries 152 of these: the QK-norm weights, four per
    double block and two per single block.

    A 2-D weight left at high precision is a different matter and worth looking
    at, since that one really is a Linear the recipe declined to quantize.
    """
    wanted = set(modules)
    ranks: Counter = Counter()
    for record in tensors.values():
        if record["component"] != component or not record["name"].endswith(".weight"):
            continue
        if _module_of(record["name"]) in wanted:
            ranks["1-D normalization vectors" if len(record["shape"]) < 2 else "2-D linear"] += 1
    return ranks


def _coverage(states: dict[str, str]) -> dict[str, Any]:
    """Summarise how much of the component was quantized.

    Counted over weight-bearing modules rather than tensors. Each quantized
    module contributes several tensors -- packed weights plus block and tensor
    scales -- so a tensor count would report a share that rises purely because
    quantization adds tensors.
    """
    counts = Counter(states.values())
    total = sum(counts.values())
    return {
        "weight_bearing_modules": total,
        "quantized": counts.get("quantized", 0),
        "high_precision": counts.get("high-precision", 0),
        "quantized_share": round(counts.get("quantized", 0) / total, 3) if total else 0.0,
    }


def _module_table(states: dict[str, str]) -> list[dict[str, Any]]:
    """Group modules by structural path so 38 identical blocks read as one row."""
    grouped: dict[str, Counter] = defaultdict(Counter)
    for module, state in states.items():
        grouped[_collapse(module)][state] += 1

    rows = []
    for pattern, counts in grouped.items():
        rows.append(
            {
                "module": pattern,
                "quantized": counts.get("quantized", 0),
                "high_precision": counts.get("high-precision", 0),
                "depth": _depth(pattern),
            }
        )
    rows.sort(key=lambda r: (r["depth"], r["module"]))
    return rows


def write_inventory(tensors: dict[str, dict[str, Any]], destination: Path) -> int:
    """Dump every tensor with its shape and dtype, one row per tensor.

    This is the raw material for the side-by-side comparison: the full contents
    of the checkpoint, not a summary of it. Written as CSV so it opens in a
    spreadsheet and can be diffed against the published checkpoint's inventory
    without any tooling.
    """
    import csv

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["component", "tensor", "dtype", "shape", "elements"])
        for record in sorted(tensors.values(), key=lambda r: (r["component"], r["name"])):
            shape = record["shape"]
            elements = 1
            for dimension in shape:
                elements *= dimension
            writer.writerow(
                [
                    record["component"],
                    record["name"],
                    record["dtype"],
                    "x".join(str(d) for d in shape) or "scalar",
                    elements,
                ]
            )
    return len(tensors)


def _is_scale_tensor(name: str) -> bool:
    """Scaling metadata rather than a weight the network multiplies by."""
    return any(name.endswith(suffix) or name.endswith("." + suffix) for suffix in SCALE_SUFFIXES)


def precision_by_role(tensors: dict[str, dict[str, Any]], component: str) -> dict[str, Any]:
    """Count *weights* by precision, ignoring scaling metadata.

    Raw dtype histograms mislead here. An NVFP4 tensor is stored as ``U8`` with a
    companion ``F8_E4M3`` block scale, so a checkpoint with 152 NVFP4 weights
    shows 152 ``U8`` **and** 152 ``F8_E4M3`` entries. Any ``F8_E4M3`` beyond that
    is a genuine FP8 weight — a layer quantized to 8 bits rather than 4.

    That distinction is the whole comparison: it separates "we quantized the same
    layers" from "we quantized more layers, or harder". Counting weights only,
    split by precision, makes it readable directly.
    """
    buckets: Counter = Counter()
    examples: dict[str, list[str]] = defaultdict(list)

    for record in tensors.values():
        if record["component"] != component:
            continue
        name = record["name"]
        if _is_scale_tensor(name) or not name.endswith(".weight"):
            continue
        dtype = record["dtype"]
        if dtype in ("U8", "I8"):
            label = "4-bit (packed)"
        elif dtype.startswith("F8"):
            label = "8-bit float"
        else:
            label = f"high precision ({dtype})"
        buckets[label] += 1
        if len(examples[label]) < 3:
            shape = "x".join(str(d) for d in record["shape"])
            examples[label].append(f"{name} {shape}")

    total = sum(buckets.values())
    return {
        "weight_tensors": total,
        "by_precision": dict(buckets.most_common()),
        "examples": dict(examples),
    }


def _shape_signatures(tensors: dict[str, dict[str, Any]], component: str) -> Counter:
    """Count tensors by (dtype, shape), ignoring names entirely."""
    return Counter(
        (record["dtype"], tuple(record["shape"]))
        for record in tensors.values()
        if record["component"] == component
    )


def compare_shapes(
    ours: dict[str, dict[str, Any]],
    theirs: dict[str, dict[str, Any]],
    our_component: str,
    their_component: str,
) -> dict[str, Any]:
    """Compare two checkpoints by shape and dtype rather than by tensor name.

    Names cannot be used: the published checkpoint is ComfyUI format and ours is
    Diffusers, so the two vocabularies do not overlap at all. Shapes do carry
    over, because they are a property of the architecture rather than of the
    naming convention. If both checkpoints contain the same multiset of tensor
    shapes at the same precisions, they describe the same network quantized the
    same way — which is the question worth answering.

    Reported as counts per signature so a missing or extra tensor shows up
    rather than averaging away.
    """
    ours_sig = _shape_signatures(ours, our_component)
    theirs_sig = _shape_signatures(theirs, their_component)

    shared = set(ours_sig) & set(theirs_sig)
    matched = sum(min(ours_sig[s], theirs_sig[s]) for s in shared)

    def _fmt(signature):
        dtype, shape = signature
        return {"dtype": dtype, "shape": list(shape)}

    only_ours = sorted(
        (s for s in ours_sig if s not in theirs_sig),
        key=lambda s: -ours_sig[s],
    )
    only_theirs = sorted(
        (s for s in theirs_sig if s not in ours_sig),
        key=lambda s: -theirs_sig[s],
    )

    return {
        "ours_tensor_count": sum(ours_sig.values()),
        "reference_tensor_count": sum(theirs_sig.values()),
        "distinct_signatures_ours": len(ours_sig),
        "distinct_signatures_reference": len(theirs_sig),
        "shared_signatures": len(shared),
        "tensors_matched_by_signature": matched,
        "share_matched": (
            round(matched / sum(ours_sig.values()), 3) if sum(ours_sig.values()) else 0.0
        ),
        "only_in_ours": [{**_fmt(s), "count": ours_sig[s]} for s in only_ours[:15]],
        "only_in_reference": [{**_fmt(s), "count": theirs_sig[s]} for s in only_theirs[:15]],
    }


def largest_tensors(
    tensors: dict[str, dict[str, Any]], component: str, limit: int = 12
) -> list[dict[str, Any]]:
    """The biggest tensors, which is where the quantization actually shows up."""
    rows = []
    for record in tensors.values():
        if record["component"] != component:
            continue
        elements = 1
        for dimension in record["shape"]:
            elements *= dimension
        rows.append(
            {
                "tensor": record["name"],
                "dtype": record["dtype"],
                "shape": record["shape"],
                "elements": elements,
            }
        )
    rows.sort(key=lambda r: -r["elements"])
    return rows[:limit]


def _dtype_histogram(tensors: dict[str, dict[str, Any]], component: str | None = None) -> dict:
    """Count tensors by dtype, most common first.

    The quickest read on whether an export is what it claims: a static NVFP4
    transformer is mostly ``U8`` with an ``F8_E4M3`` scale per quantized layer,
    and one that is still overwhelmingly ``BF16`` never had the recipe applied.
    """
    return dict(
        Counter(
            t["dtype"]
            for t in tensors.values()
            if component is None or t["component"] == component
        ).most_common()
    )


def _pick_transformer(tensors: dict[str, dict[str, Any]]) -> str:
    """The component holding the denoiser, which is the only one we quantize."""
    components = {t["component"] for t in tensors.values()}
    for candidate in ("transformer", "unet", "root"):
        if candidate in components:
            return candidate
    return max(components, key=lambda c: sum(1 for t in tensors.values() if t["component"] == c))


def run(*, workspace, config_path: Path, manifest, args) -> dict[str, Any]:
    """Report which layers carry 4-bit weights, and whether the filter agrees.

    Reads dtypes straight out of the safetensors headers -- no weights are ever
    loaded, so this runs on a login node without a GPU -- and writes
    ``results/schema_diff-<model>.json`` plus a per-tensor inventory CSV. Needs
    the export stage.

    The authoritative check imports Model Optimizer's real ``filter_func`` and
    evaluates it against every module found. If that import fails the stage still
    completes and still reports dtypes, but says the filter was not checked. That
    line is worth reading: a run without the filter check looks almost identical
    to one with it and proves considerably less.
    """
    config = json.loads(config_path.read_text())

    ours_dir = workspace.exports / paths.export_dir_name(config) / "hf"

    # An absent or empty ``reference_dir`` means there is no published checkpoint
    # to compare against -- flux-schnell has none, and its configs carry "".
    # Falling through to ``workspace.models / ""`` would resolve to the models
    # root, whose BF16 baselines would then be read as the published NVFP4
    # reference and reported as one: a full comparison table, entirely spurious.
    reference_name = config.get("reference_dir") or None
    reference_dir = (workspace.models / reference_name) if reference_name else None

    if getattr(args, "dry_run", False):
        target = reference_dir if reference_dir is not None else "(no published reference)"
        print(f"  would inspect {ours_dir} and compare against {target}")
        return {}

    if not ours_dir.exists():
        raise RuntimeError(f"No export at {ours_dir}. Run the export stage first.")

    tensors = _read_headers(ours_dir)
    if not tensors:
        raise RuntimeError(f"No safetensors found under {ours_dir}.")

    component = _pick_transformer(tensors)
    states = classify_modules(tensors, component)
    coverage = _coverage(states)
    components = sorted({t["component"] for t in tensors.values()})

    print(f"  {len(tensors)} tensors across {len(components)} components: {', '.join(components)}")
    print(f"  quantizing '{component}' — everything else stays at high precision by design")
    print(f"  dtypes in {component}: {_dtype_histogram(tensors, component)}")
    print(
        f"  {coverage['quantized']} of {coverage['weight_bearing_modules']} weight-bearing "
        f"modules carry 4-bit weights ({coverage['quantized_share']:.0%})"
    )

    report: dict[str, Any] = {
        "export_path": str(ours_dir),
        "components": components,
        "quantized_component": component,
        "tensor_count": len(tensors),
        "dtype_histogram_all": _dtype_histogram(tensors),
        "dtype_histogram_transformer": _dtype_histogram(tensors, component),
        "coverage": coverage,
        "modules": _module_table(states),
    }

    # The authoritative check: run the export's own filter over what we produced.
    filter_func, provenance = _load_modelopt_filter()
    report["filter_source"] = provenance

    if filter_func is not None:
        agreement = check_filter_agreement(states, filter_func)
        report["filter_agreement"] = agreement
        print(f"\n  exclusion filter: {provenance}")
        if agreement["error_count"]:
            print(
                f"  INCONCLUSIVE: the filter raised on {agreement['error_count']} of "
                f"{agreement['total_modules']} modules, so agreement cannot be claimed:"
            )
            for err in agreement["errors"]:
                print(f"    {err}")
        elif agreement["agrees"]:
            print(
                f"  VERIFIED: every layer the filter excludes is at high precision "
                f"({agreement['checked']} modules checked)"
            )
        else:
            print(
                f"  DEFECT: {len(agreement['excluded_but_quantized'])} layers the filter "
                "excludes carry 4-bit weights:"
            )
            for module in agreement["excluded_but_quantized"][:10]:
                print(f"    {module}")
        extra = agreement["included_but_high_precision"]
        if extra:
            ranks = rank_breakdown(tensors, component, extra)
            report["permitted_but_high_precision"] = dict(ranks)
            print(f"  {len(extra)} modules the filter permits were left at high precision:")
            for label, count in sorted(ranks.items()):
                print(f"    {count:>4}  {label}")
            if ranks.get("1-D normalization vectors"):
                print(
                    "    1-D weights are normalization scale vectors, not matrix "
                    "multiplies, so there is nothing to accelerate. Expected."
                )
            if ranks.get("2-D linear"):
                print(
                    "    2-D weights left at high precision are real Linear layers "
                    "the recipe declined to quantize. Worth checking."
                )
    else:
        print(f"\n  exclusion filter not checked: {provenance}")
        print("  Set MODELOPT_DIFFUSERS_DIR to a Model-Optimizer checkout for a definitive answer.")

    # Name-based reporting, kept separate and clearly weaker than the filter check.
    exclusions = tuple(config.get("expected_exclusions") or DOCUMENTED_EXCLUSIONS)
    names = exclusion_report(states, exclusions)
    report["documented_exclusions"] = {"names": list(exclusions), **names}

    if names["top_level"]:
        print("\n  top-level modules matching the documented exclusion list:")
        for module, state in names["top_level"].items():
            flag = "  <-- expected high precision" if state == "quantized" else ""
            print(f"    {module:<40} {state}{flag}")
    if names["nested"]:
        print("\n  same leaf names inside repeated blocks — these are ordinary GEMMs:")
        for pattern, counts in names["nested"].items():
            summary = ", ".join(f"{v} {k}" for k, v in sorted(counts.items()))
            print(f"    {pattern:<40} {summary}")

    # Full tensor dump. The raw two-column material, not a summary of it.
    # Per-model filenames. Fixed names meant each model's schema run silently
    # overwrote the last one, so after running all three arms only the most
    # recent survived -- and nothing in the file said which model it described.
    # Named model_label, not label: the precision-comparison loop below does
    # `for label in labels`, which silently rebound a plainer name and produced
    # `schema_diff-high precision (BF16).json`.
    model_label = paths.export_dir_name(config)
    inventory = workspace.results / f"tensor_inventory_ours-{model_label}.csv"
    write_inventory(tensors, inventory)
    report["inventory_ours"] = str(inventory)
    print(f"\n  every tensor with shape and dtype: {inventory}")

    print(f"\n  largest tensors in {component}:")
    for row in largest_tensors(tensors, component):
        shape = "x".join(str(d) for d in row["shape"])
        print(f"    {row['dtype']:<9} {shape:<20} {row['tensor']}")

    # Comparison against the published checkpoint, matched on shape rather than
    # name. This is the check that decides whether the export is servable.
    have_reference = (
        reference_dir is not None
        and reference_dir.exists()
        and any(reference_dir.rglob("*.safetensors"))
    )
    # The no-reference case is handled once, in the else of this branch. It used
    # to be written twice -- a standalone `if not have_reference` here that set
    # report["reference"] = None and printed one message, and the else below that
    # set the same key and printed a different one. Both ran, so the output said
    # "comparison skipped" and then "reporting our export only" for the same
    # condition.
    if have_reference:
        theirs = _read_headers(reference_dir)
        their_component = _pick_transformer(theirs)

        their_inventory = workspace.results / f"tensor_inventory_reference-{model_label}.csv"
        write_inventory(theirs, their_inventory)

        shapes = compare_shapes(tensors, theirs, component, their_component)
        our_roles = precision_by_role(tensors, component)
        their_roles = precision_by_role(theirs, their_component)
        report["precision_by_role"] = {"ours": our_roles, "reference": their_roles}
        report["reference"] = {
            "path": str(reference_dir),
            "inventory": str(their_inventory),
            "tensor_count": len(theirs),
            "dtype_histogram": _dtype_histogram(theirs, their_component),
            "shape_comparison": shapes,
            "caveat": "Matched on (dtype, shape), never on tensor name: the published "
            "checkpoint is ComfyUI format and ours is Diffusers, so the naming "
            "vocabularies do not overlap. Shapes are a property of the architecture and "
            "do carry across formats. Totals still differ because our export includes "
            "the text encoders and VAE, so ratios of whole-checkpoint counts are "
            "meaningless; the transformer comparison below is the real signal.",
        }

        print(f"\n  published reference: {their_inventory}")
        print(f"  dtypes in their {their_component}: {_dtype_histogram(theirs, their_component)}")

        # The comparison that actually answers "did we quantize the same layers".
        print("\n  weights by precision — scaling metadata excluded:")
        labels = sorted(set(our_roles["by_precision"]) | set(their_roles["by_precision"]))
        print(f"    {'precision':<26} {'ours':>8} {'published':>10}")
        for label in labels:
            print(
                f"    {label:<26} {our_roles['by_precision'].get(label, 0):>8} "
                f"{their_roles['by_precision'].get(label, 0):>10}"
            )
        their_fp8 = their_roles["by_precision"].get("8-bit float", 0)
        our_fp8 = our_roles["by_precision"].get("8-bit float", 0)
        if their_fp8 > our_fp8:
            print(
                f"\n  The published checkpoint keeps {their_fp8 - our_fp8} more layers at FP8 "
                "than we do, meaning we quantized them harder. Examples of theirs:"
            )
            for example in their_roles["examples"].get("8-bit float", []):
                print(f"    {example}")
            print(
                "  Worth raising: the public filter does not protect these, so the published "
                "recipe is not simply the public defaults."
            )
        elif our_fp8 > their_fp8:
            print(
                f"\n  We keep {our_fp8 - their_fp8} more layers at FP8 than the published "
                "checkpoint, so our recipe is the more conservative one."
            )
        print(
            f"  shape match: {shapes['tensors_matched_by_signature']} of "
            f"{shapes['ours_tensor_count']} of our transformer tensors have a "
            f"(dtype, shape) twin in theirs ({shapes['share_matched']:.0%})"
        )
        print(
            f"  {shapes['shared_signatures']} shared signatures; "
            f"{shapes['distinct_signatures_ours']} distinct in ours, "
            f"{shapes['distinct_signatures_reference']} in theirs"
        )
        if shapes["only_in_ours"]:
            print("  shapes only in ours:")
            for row in shapes["only_in_ours"][:5]:
                print(f"    {row['count']:>4}x  {row['dtype']:<9} {row['shape']}")
        if shapes["only_in_reference"]:
            print("  shapes only in theirs:")
            for row in shapes["only_in_reference"][:5]:
                print(f"    {row['count']:>4}x  {row['dtype']:<9} {row['shape']}")
        print(
            "\n  Read this as: do we describe the same network at the same precisions. "
            "Names will not line up across formats and are not expected to."
        )
    else:
        report["reference"] = None
        print("\n  no published reference downloaded; reporting our export only")
        print("  Download it to get the golden-standard shape comparison.")

    report["model"] = model_label
    out_path = workspace.results / f"schema_diff-{model_label}.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n")

    # Unsuffixed copy for anything that looks for a fixed name. It is always the
    # most recent run, which is why the suffixed file above is the one to quote.
    (workspace.results / "schema_diff.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"\n  written to {out_path}")

    summary = {"report": str(out_path), "coverage": coverage, "filter_source": provenance}
    if "filter_agreement" in report:
        summary["filter_agrees"] = report["filter_agreement"]["agrees"]
    return summary
