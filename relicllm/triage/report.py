"""Assemble the four gates into one answer, and print it as a table or as JSON.

Everything the report says is something a gate produced, and every number in it carries how it was
arrived at. That is the whole design: this module computes nothing. Its job is to put the *upgrade*
next to the tier, because the tier alone is the least useful thing the tool knows.

An example of why. At an 8k input the roster's three fastest models measure 1819, 753 and 639 tok/s,
and all three have no serving path at all; the fastest model that *is* served sits far below them. A
report that printed "DEMO" for each of the three would be technically correct and would have thrown
away the finding. So the verdict carries ``upgrade`` -- what a person would have to build to move it
up a tier -- and the reachability notes beside it.

The other thing this module refuses to do is round an absence into a number. A measurement that was
not taken yields ``CANDIDATE``, a host memory that was not stated yields a note, a conversion rule
that does not exist yields the file's bytes *and* the statement that they may be too small. A
report that guesses well is worse than one that says what it does not know, because only the second
can be acted on.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from relicllm.components.moe.placement import HardwareProfile, usable_per_card
from relicllm.triage.checkpoint import CheckpointInventory, read_inventory
from relicllm.triage.fit import DEFAULT_BATCHES, DEFAULT_CONTEXTS, FitResult, fit_all, hardware_profile_for
from relicllm.triage.kv import KvGeometry, kv_geometry
from relicllm.triage.reach import Reachability, reachability
from relicllm.triage.tier import DEFAULT_THRESHOLDS, Measured, Thresholds, Tier, Verdict, tier
from relicllm.triage.weights import PrecisionOption, precision_ladder

__all__ = ["Assessment", "assess", "hardware_profile_for"]


@dataclass(frozen=True)
class Assessment:
    """One checkpoint, four gates, one answer."""

    checkpoint: str
    architecture: str | None
    inventory: CheckpointInventory
    geometry: KvGeometry
    options: tuple[PrecisionOption, ...]
    fits: tuple[FitResult, ...]
    reach: Reachability
    verdict: Verdict
    hardware: HardwareProfile
    thresholds: Thresholds
    context: int
    batch: int
    reserve_fraction: float
    measured: Measured | None = None
    unreadable: str | None = None
    """Set when the checkpoint's headers could not be read at all; the rest is then minimal."""

    @property
    def best_fit(self) -> FitResult | None:
        """The cheapest placement that exists: fewest cards, then fewest bytes on a card."""
        fitting = [fit for fit in self.fits if fit.fits]
        if not fitting:
            return None
        return min(fitting, key=lambda fit: (fit.min_cards or 0, fit.weights_per_rank_bytes))

    def upgrade(self) -> str:
        """What would move this answer up a tier, in the order that costs least.

        A one-liner rather than a plan, and it names the gate rather than the symptom: "measure it"
        and "write a kernel" are different people's work, and which one applies is the thing a reader
        cannot tell from the tier.
        """
        if self.unreadable:
            return f"the checkpoint's headers do not read: {self.unreadable}"
        if self.verdict.tier is Tier.IMPOSSIBLE:
            best = min(self.fits, key=lambda fit: fit.weights_per_rank_bytes, default=None)
            if best is None:
                return "nothing: no precision of this checkpoint fits this box"
            return (
                f"nothing at this precision set: the smallest placement is "
                f"{best.weights_per_rank_gib:.1f} GiB per card at {best.cards_at_width} cards plus "
                f"{best.kv_bytes_per_token_per_rank / 1024 ** 2:.1f} MiB of KV per token per rank, "
                "and a GGUF or FP4 release with a smaller per-token cache is the only route"
            )
        if self.verdict.tier is Tier.CANDIDATE:
            return (
                "measure it: " + self._probe_command()
                + " -- then re-run with --measured"
            )
        if self.verdict.tier is Tier.DEMO:
            return (
                "no threshold moves this without a faster runtime; the gap is "
                + self.verdict.reason
            )
        if self.verdict.tier is Tier.CHAT:
            return (
                f"production needs {self.thresholds.prefill_8k_tok_per_second:g} tok/s of aggregate "
                f"prefill at input={self.thresholds.input_len}; batching is the lever, and "
                "supports_batch is false for every runtime in the tree today"
            )
        return "nothing: both SLOs are met as measured"

    def _probe_command(self) -> str:
        """The measuring run a human would pay for, as the two commands it actually takes.

        Built from **this repository's own flags**, which is the whole point of printing it: a probe
        that does not run is worse than no probe, because it reads as validated. So the server is
        started with ``relicllm.cli``'s ``serve`` subcommand and the client is
        ``tests/bench_serving.py`` with its real flag names -- ``--random-input-len``, not
        ``--input-len`` -- and with the SLOs handed to ``--goodput`` in the vLLM vocabulary
        ``docs/guides/latency_metrics.md`` defines, so the run reports goodput rather than only
        latency.
        """
        if self.reach.adapter is None:
            return (
                "no runnable probe: nothing declared in RUNTIMES reads this architecture, so there "
                "is no `serve` command to measure through yet"
            )
        checkpoint = os.path.abspath(self.checkpoint)
        return (
            f"CUDA_VISIBLE_DEVICES=$(seq -s, 0 {self.hardware.gpu_count - 1}) "
            f"/home/lvyufeng/miniconda3/envs/deepseek/bin/python -m relicllm.cli serve "
            f"--model {checkpoint} --backend {self.reach.adapter} "
            f"--tensor-parallel-size {self.best_fit.min_cards if self.best_fit else 1} "
            f"--max-model-len {max(self.context, self.thresholds.input_len)}"
            " &  then  "
            f"/home/lvyufeng/miniconda3/envs/deepseek/bin/python tests/bench_serving.py "
            f"--random-input-len {self.thresholds.input_len} "
            f"--random-output-len {self.thresholds.output_len} "
            f"--goodput ttft:{self.thresholds.ttft_1k_seconds * 1000:.0f} "
            f"tpot:{self.thresholds.tpot_seconds * 1000:.0f} --json-out results.json"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint": self.checkpoint,
            "architecture": self.architecture,
            "format": self.inventory.format if not self.unreadable else "unreadable",
            "unreadable": self.unreadable,
            "tier": str(self.verdict.tier),
            "reason": self.verdict.reason,
            "decided_by": self.verdict.decided_by,
            "unknown": list(self.verdict.unknown),
            "upgrade": self.upgrade(),
            "thresholds": {
                "basis": self.thresholds.basis,
                "ttft_1k_seconds": self.thresholds.ttft_1k_seconds,
                "tpot_seconds": self.thresholds.tpot_seconds,
                "prefill_8k_tok_per_second": self.thresholds.prefill_8k_tok_per_second,
            },
            "hardware": {
                "name": self.hardware.name,
                "gpu_count": self.hardware.gpu_count,
                "gpu_memory_gib": self.hardware.gpu_memory_gib,
                "host_memory_gib": self.hardware.host_memory_gib,
                "usable_per_card_gib": usable_per_card(
                    self.hardware, reserve_fraction=self.reserve_fraction
                )
                / 1024 ** 3,
                "reserve_fraction": self.reserve_fraction,
            },
            "weights": {
                "total_gib": self.inventory.total_bytes / 1024 ** 3,
                "resident_gib": self.inventory.resident_bytes / 1024 ** 3,
                "offloadable_gib": self.inventory.offloadable_bytes / 1024 ** 3,
                "tensor_count": self.inventory.tensor_count,
                "parameter_count": self.inventory.parameter_count,
                "bytes_by_role": {k: v for k, v in sorted(self.inventory.bytes_by_role.items())},
                "bytes_by_dtype": {k: v for k, v in sorted(self.inventory.bytes_by_dtype.items())},
            },
            "kv": {
                "attention_kind": self.geometry.attention_kind,
                "sharding": self.geometry.sharding,
                "sharding_source": self.geometry.sharding_source,
                "preallocated": self.geometry.preallocated,
                "values_per_token_per_layer": self.geometry.values_per_token_per_layer,
                "bytes_per_token_whole_model": self.geometry.bytes_per_token_whole_model,
                "confidence": str(self.geometry.confidence),
                "layers": [
                    {
                        "count": layer.count,
                        "kind": layer.kind,
                        "values_per_token": layer.values_per_token,
                        "compress_ratio": layer.compress_ratio,
                        "source": layer.source,
                        "note": layer.note,
                    }
                    for layer in self.geometry.layers
                ],
                "notes": list(self.geometry.notes),
                "sources": list(self.geometry.sources),
            },
            "precision_ladder": [
                {
                    "label": option.label,
                    "path": option.path,
                    "format": option.format,
                    "precision": option.precision,
                    "bytes_total": option.bytes_total,
                    "on_card_bytes": option.on_card_bytes,
                    "resident_on_card_bytes": option.resident_on_card_bytes,
                    "confidence": str(option.confidence),
                    "evidence": option.evidence,
                    "basis": list(option.basis),
                    "unknown": list(option.unknown),
                }
                for option in self.options
            ],
            "fits": [
                {
                    "label": fit.label,
                    "fits": fit.fits,
                    "min_cards": fit.min_cards,
                    "cards_at_width": fit.cards_at_width,
                    "binding_constraint": fit.binding_constraint,
                    "weights_per_rank_bytes": fit.weights_per_rank_bytes,
                    "kv_bytes_per_token_per_rank": fit.kv_bytes_per_token_per_rank,
                    "host_bank_gib": fit.host_bank_gib,
                    "context": fit.context,
                    "batch": fit.batch,
                    "curve": [
                        {
                            "context": point.context,
                            "batch": point.max_batch,
                            "kv_bytes_per_rank": point.kv_bytes_per_rank,
                            "headroom_bytes": point.headroom_bytes,
                        }
                        for point in fit.curve.points
                    ],
                    "basis": list(fit.basis),
                    "notes": list(fit.notes),
                    "unknown": list(fit.unknown),
                }
                for fit in self.fits
            ],
            "reachability": {
                "adapter": self.reach.adapter,
                "adapter_reason": self.reach.adapter_reason,
                "loader": self.reach.loader,
                "missing_kernel_types": list(self.reach.missing_kernel_types),
                "work_items": list(self.reach.work_items()),
                "notes": list(self.reach.notes),
            },
        }

    def format_text(self, *, width: int = 100) -> str:
        """The human table. Fixed-width, no colour, and every budget printed with its use."""
        lines: list[str] = []
        add = lines.append

        add(f"{os.path.basename(os.path.abspath(self.checkpoint))}  [{self.architecture or 'unknown architecture'}]")
        add("=" * width)
        if self.unreadable:
            add(f"  UNREADABLE  {self.unreadable}")
            add("")
            add("  upgrade: " + self.upgrade())
            return "\n".join(lines)

        tier_upper = str(self.verdict.tier).upper()
        add(f"  {tier_upper}  --  {self.verdict.reason}")
        add(f"  decided by: {self.verdict.decided_by}")
        if self.verdict.unknown:
            add(f"  waiting on: {', '.join(self.verdict.unknown)}")
        add("")

        add("  weights")
        add(f"    {self.inventory.tensor_count} tensors, {self.inventory.parameter_count / 1e9:.1f}B parameters")
        add(f"    {self.inventory.total_bytes / 1024 ** 3:8.2f} GiB on disk  "
            f"{self.inventory.resident_bytes / 1024 ** 3:8.2f} resident  "
            f"{self.inventory.offloadable_bytes / 1024 ** 3:8.2f} offloadable")
        for role, size in sorted(self.inventory.bytes_by_role.items(), key=lambda item: -item[1]):
            add(f"      {role:16s} {size / 1024 ** 3:8.2f} GiB")
        # The section above is the *file*; the fit below is the *card*. They are the same numbers on
        # a checkpoint nothing converts, and they are 146 GiB apart on DeepSeek-V4-Flash, whose
        # packed FP4 experts are materialized to INT8 before anything runs. Printing both without the
        # rules that separate them would read as the report contradicting itself.
        conversions = [line for option in self.options for line in option.basis]
        if conversions:
            add(f"    on a card: {self.options[0].on_card_bytes / 1024 ** 3:.2f} GiB after")
            for line in dict.fromkeys(conversions):
                add(f"      {line}")
        add("")

        add("  kv cache")
        add(f"    {self.geometry.attention_kind}, {self.geometry.sharding}, "
            f"{'preallocated' if self.geometry.preallocated else 'per request'}")
        add(f"    {self.geometry.bytes_per_token_whole_model} B/token whole-model   "
            f"[{self.geometry.confidence}]")
        for layer in self.geometry.layers:
            ratio = f" ratio {layer.compress_ratio}" if layer.compress_ratio else ""
            add(f"      {layer.count:3d} x {layer.kind:15s} {layer.values_per_token:6d} values/token{ratio}")
        for note in self.geometry.notes:
            add(f"      note: {note}")
        add("")

        add("  fit")
        add(f"    {self.hardware.name}: {self.hardware.gpu_count} x {self.hardware.gpu_memory_gib:g} GiB, "
            f"reserve {self.reserve_fraction:g} -> "
            f"{usable_per_card(self.hardware, reserve_fraction=self.reserve_fraction) / 1024 ** 3:.1f} GiB/card usable")
        if self.hardware.knows_host_memory:
            add(f"    host memory {self.hardware.host_memory_gib:g} GiB")
        # The width asked for, not the width used: a single-card runtime overrides it to 1, and the
        # card count on each fit line below is what the fit was actually priced at.
        for fit in self.fits:
            verdict = f"{fit.min_cards} card(s)" if fit.fits else "does not fit"
            # The bank is printed beside the resident weights because the two precisions of one
            # model often differ *only* there: DeepSeek-V4-Flash's w8a8 build is 1.835x the native
            # checkpoint's bytes and identical on a card, and a row that showed one number would
            # make the ladder look like three copies of the same artifact.
            bank = f" + {fit.host_bank_gib:6.2f} GiB bank" if fit.host_bank_gib else ""
            add(f"    {fit.label:20s} {verdict:14s} {fit.binding_constraint:10s} "
                f"weights/rank {fit.weights_per_rank_gib:6.2f} GiB{bank}")
            for note in fit.notes:
                add(f"        {note}")
        best = self.best_fit
        if best is not None:
            add("")
            add(f"    best placement: {best.min_cards} card(s), context x batch headroom")
            for point in best.curve.points:
                mark = "ok " if point.headroom_bytes >= 0 else "no "
                add(f"      {mark} L={point.context:7d} B={point.max_batch:2d}  "
                    f"kv {point.kv_bytes_per_rank / 1024 ** 3:7.3f} GiB/rank  "
                    f"headroom {point.headroom_bytes / 1024 ** 3:8.2f} GiB")
        add("")

        add("  reachability (never affects the tier)")
        add(f"    adapter: {self.reach.adapter or 'none'}")
        for item in self.reach.work_items():
            add(f"      work: {item}")
        for note in self.reach.notes:
            add(f"      note: {note}")
        add("")

        unknown = sorted({entry for fit in self.fits for entry in fit.unknown})
        if unknown:
            add("  unknown")
            for entry in unknown:
                add(f"    {entry}")
            add("")

        add(f"  upgrade: {self.upgrade()}")
        return "\n".join(lines)


def assess(
    checkpoint: str,
    *,
    hardware: HardwareProfile | None = None,
    measured: Measured | None = None,
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
    context: int = 8192,
    batch: int = 1,
    reserve_fraction: float = 0.15,
    host_reserve_fraction: float = 0.10,
    workspace_bytes: int = 0,
    contexts: tuple[int, ...] = DEFAULT_CONTEXTS,
    batches: tuple[int, ...] = DEFAULT_BATCHES,
    include_siblings: bool = True,
) -> Assessment:
    """Run all four gates on one checkpoint.

    The order is the cost order and it is not an optimisation: gates 0 and 1 are arithmetic over
    headers, so they run always, and gate 2 is never run here at all. ``measured`` is passed *in* --
    the tool prints the command that would produce it and stops, because a run costs GPU-hours and
    that is the caller's decision to make, not a default to be taken.
    """
    hardware = hardware or hardware_profile_for()
    try:
        inventory = read_inventory(checkpoint)
    except (FileNotFoundError, ValueError, OSError) as error:
        return _unreadable_assessment(
            checkpoint, hardware=hardware, thresholds=thresholds, context=context, batch=batch,
            reserve_fraction=reserve_fraction, error=error, measured=measured,
        )

    geometry = kv_geometry(
        inventory.config,
        architecture=inventory.architecture,
        tensor_names=inventory.tensor_names,
    )
    options = precision_ladder(checkpoint, include_siblings=include_siblings)
    fits = fit_all(
        inventory=inventory,
        options=options,
        geometry=geometry,
        hardware=hardware,
        context=context,
        batch=batch,
        reserve_fraction=reserve_fraction,
        host_reserve_fraction=host_reserve_fraction,
        workspace_bytes=workspace_bytes,
        contexts=contexts,
        batches=batches,
    )
    reaches = reachability(inventory)
    verdict = tier(fits=any(fit.fits for fit in fits), measured=measured, thresholds=thresholds)
    return Assessment(
        checkpoint=os.path.abspath(checkpoint),
        architecture=inventory.architecture,
        inventory=inventory,
        geometry=geometry,
        options=options,
        fits=fits,
        reach=reaches,
        verdict=verdict,
        hardware=hardware,
        thresholds=thresholds,
        context=context,
        batch=batch,
        reserve_fraction=reserve_fraction,
        measured=measured,
    )


def _unreadable_assessment(
    checkpoint: str,
    *,
    hardware: HardwareProfile,
    thresholds: Thresholds,
    context: int,
    batch: int,
    reserve_fraction: float,
    error: BaseException,
    measured: Measured | None,
) -> Assessment:
    """An assessment for a checkpoint that could not be read, which is *not* the same as one that
    does not fit.

    A missing shard, a corrupt header and a directory that is not a checkpoint at all are all
    "unreadable", and reporting any of them as ``IMPOSSIBLE`` would be a claim about the model made
    from a claim about the disk. Gate 0 has not run, so nothing has been disproved -- the verdict is
    ``CANDIDATE``, which is exactly what "not yet known" is for.

    **The measurement is dropped, not carried.** A caller that supplied one gets ``CANDIDATE`` back
    and not a tier, because a tier is a statement about a checkpoint and this is a statement about a
    disk. Carrying it through would let a complete ``--measured`` file turn an unread checkpoint into
    ``PRODUCTION`` in the JSON, while ``--brief`` and the text report -- which both short-circuit to
    UNREADABLE -- said otherwise. ``--measured`` is one file applied to every checkpoint on a
    multi-checkpoint run, so a tier here can come from another model's measurement entirely.
    """
    measured = None
    empty_inventory = CheckpointInventory(
        path=os.path.abspath(checkpoint),
        format="unreadable",
        architecture=None,
        config={},
        tensor_names=(),
        elements_by_tensor={},
        bytes_by_role={},
        bytes_by_role_and_dtype={},
        bytes_by_dtype={},
        tensor_count=0,
        total_bytes=0,
        unknown=(f"{type(error).__name__}: {error}",),
    )
    geometry = kv_geometry({})
    verdict = tier(fits=True, measured=measured, thresholds=thresholds)
    return Assessment(
        checkpoint=os.path.abspath(checkpoint),
        architecture=None,
        inventory=empty_inventory,
        geometry=geometry,
        options=(),
        fits=(),
        reach=reachability(empty_inventory),
        verdict=verdict,
        hardware=hardware,
        thresholds=thresholds,
        context=context,
        batch=batch,
        reserve_fraction=reserve_fraction,
        measured=measured,
        unreadable=f"{type(error).__name__}: {error}",
    )


def dumps(assessment: Assessment) -> str:
    return json.dumps(assessment.to_dict(), indent=2, sort_keys=False)
