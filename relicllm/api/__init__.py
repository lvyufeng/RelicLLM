"""Public backend-neutral RelicLLM API types."""

from .backend import EngineBackend
from .errors import (
    BackendUnavailableError,
    ConfigurationError,
    RelicLLMError,
    RequestCancelledError,
    TensorParallelSupervisorError,
    UnsupportedFeatureError,
)
from .types import (
    BackendCapabilities,
    EngineArgs,
    GenerationRequest,
    GenerationResult,
    HealthStatus,
    SamplingParams,
    TimingMetrics,
    TokenEvent,
    Usage,
    backend_hint,
    device_hint,
)

__all__ = [
    "BackendCapabilities",
    "BackendUnavailableError",
    "ConfigurationError",
    "EngineArgs",
    "EngineBackend",
    "GenerationRequest",
    "GenerationResult",
    "HealthStatus",
    "RelicLLMError",
    "RequestCancelledError",
    "SamplingParams",
    "TensorParallelSupervisorError",
    "TimingMetrics",
    "TokenEvent",
    "backend_hint",
    "device_hint",
    "UnsupportedFeatureError",
    "Usage",
]
