"""Common MoE model spec framework."""

from relicllm.components.moe.registry import detect_spec, get_spec, known_architectures
from relicllm.components.moe.spec import (
    CapabilityItem,
    CapabilityReport,
    MoEArchitectureParams,
    MoEModelSpec,
    PlacementDecision,
    SpecValidation,
    TensorMapping,
)

__all__ = [
    "CapabilityItem",
    "CapabilityReport",
    "MoEArchitectureParams",
    "MoEModelSpec",
    "PlacementDecision",
    "SpecValidation",
    "TensorMapping",
    "detect_spec",
    "get_spec",
    "known_architectures",
]
