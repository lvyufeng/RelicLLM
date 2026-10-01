"""Smoke test to verify basic import and API availability after installation."""

import sys


def test_import_relicllm():
    """Test that relicllm can be imported."""
    import relicllm
    assert relicllm.__version__ is not None


def test_public_api_available():
    """Test that public API classes are accessible."""
    import relicllm

    # Core API classes
    assert hasattr(relicllm, 'LLM')
    assert hasattr(relicllm, 'AsyncLLM')
    assert hasattr(relicllm, 'EngineArgs')
    assert hasattr(relicllm, 'SamplingParams')
    assert hasattr(relicllm, 'GenerationRequest')
    assert hasattr(relicllm, 'GenerationResult')

    # Exception classes
    assert hasattr(relicllm, 'RelicLLMError')
    assert hasattr(relicllm, 'BackendUnavailableError')
    assert hasattr(relicllm, 'ConfigurationError')
    assert hasattr(relicllm, 'UnsupportedFeatureError')


def test_backends_importable():
    """Test that backend modules can be imported."""
    import relicllm.backends
    assert relicllm.backends is not None


def test_cli_importable():
    """Test that CLI module can be imported."""
    import relicllm.cli
    assert hasattr(relicllm.cli, 'main')


def test_version_format():
    """Test that version string has expected format."""
    import relicllm
    version = relicllm.__version__
    assert isinstance(version, str)
    parts = version.split('.')
    assert len(parts) >= 2, f"Version '{version}' should have at least major.minor"


if __name__ == '__main__':
    # Run all test functions
    test_import_relicllm()
    test_public_api_available()
    test_backends_importable()
    test_cli_importable()
    test_version_format()
    print("✅ All smoke tests passed")
