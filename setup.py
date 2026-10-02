"""Package relicllm.

Pure Python: the native kernels moved to relic-core, so there is no ext_modules
here. What remains is the runtime -- the serving adapters, the model
implementations, the loaders, and the CLI they all sit behind.
"""

from setuptools import find_namespace_packages, setup

setup(
    # One package. Resolved from the tree rather than hand-listed, so the list
    # cannot drift.
    packages=find_namespace_packages(include=["relicllm", "relicllm.*"]),
)