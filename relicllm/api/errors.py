"""Public exceptions raised by the RelicLLM API."""

from __future__ import annotations


class RelicLLMError(RuntimeError):
    """Base class for errors that are safe to expose to API clients."""


class ConfigurationError(RelicLLMError, ValueError):
    """The engine or request configuration is invalid."""


class BackendUnavailableError(RelicLLMError):
    """A requested backend is not installed or cannot be initialized."""


class UnsupportedFeatureError(RelicLLMError, NotImplementedError):
    """The selected backend does not implement a requested feature."""


class RequestCancelledError(RelicLLMError):
    """Generation was cancelled before it completed."""


class TensorParallelSupervisorError(RelicLLMError):
    """A supervised tensor-parallel rank failed to start or remain alive."""
