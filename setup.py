"""Package relicllm.

Pure Python: the native kernels moved to relic-core, so there is no ext_modules
here. What remains is the runtime -- the model implementations, the serving
adapters, and the `src` tree they import from.
"""

from setuptools import find_namespace_packages, setup

setup(
    # RelicLLM's own package plus the shared `src` tree it carries. Resolved from
    # the tree rather than hand-listed, so it cannot drift.
    packages=find_namespace_packages(
        include=["relicllm", "relicllm.*", "src", "src.*"],
        exclude=["src.csrc", "src.csrc.*", "src.gguf", "src.moe"],
    ),
)
