"""Produce the private Triton-Ascend 3.2.0 / CANN 9.x host-driver compatibility patch.

Why this exists
---------------
``triton/backends/ascend/npu_utils.cpp`` builds a name -> ``rtLimitType_t`` map for
``set_device_limit``. One entry names ``RT_LIMIT_TYPE_SIMT_WARP_STACK_SIZE``, which the
installed CANN headers do not declare:

* CANN 9.1.0 ``aarch64-linux/pkg_inc/runtime/rt_external_base.h``
  sha256 e4f372e3a3804d8061f57f12714c415f8c5ea9d34905f9c726ecdfc0ae9696c6
* CANN 9.2.0 ``aarch64-linux/pkg_inc/runtime/rt_external_base.h``
  sha256 0b89cd3b978decfc25950509f8bb162ab1c3212631ff8d44e4734df29e2cb499

Both declare ``RT_LIMIT_TYPE_SIMT_STACK_SIZE = 1`` and
``RT_LIMIT_TYPE_SIMT_DVG_WARP_STACK_SIZE = 2`` and no ``*_SIMT_WARP_STACK_SIZE``.
The compiler therefore rejects the initializer list, so the generated NPU host
extension never builds and no kernel can start. This is a host-driver bootstrap
defect, not a kernel defect.

What the patch changes, and what it deliberately does not
--------------------------------------------------------
1. Drops only that one map entry. The other three mappings are untouched, so
   ``LOW_POWER_TIMEOUT``, ``DVG_WARP_STACK_SIZE`` and ``STACK_SIZE`` behave exactly
   as before.
2. Replaces the silent ``printf`` + ``return NULL`` fallback for this name with an
   explicit ``NotImplementedError`` naming the missing feature. Returning ``NULL``
   without setting an exception would surface as an opaque ``SystemError``; the
   unsupported feature must stay *unavailable*, not appear to work.
3. Does **not** substitute ``RT_LIMIT_TYPE_SIMT_STACK_SIZE`` for the missing
   enumerator. The two names are plausibly a rename, but nothing in the installed
   headers or the package states that equivalence, and inventing it would change
   device-limit semantics silently. Mapping it is left to a future CANN that
   declares the symbol.
4. Does not touch CANN, the driver, shared headers, any kernel, torch_npu, or any
   other file in the environment.

No call path in the package uses this limit: a repository-wide search finds only the
definition, the docstring and the method table entry, so removing the mapping cannot
alter 910B execution.

The patched file is written beside this script; the installed file is not modified
here.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ORIGINAL = HERE / "npu_utils.cpp.orig"
PATCHED = HERE / "npu_utils.cpp.patched"
ORIGINAL_SHA256 = "2d6e58513486c70e47a8a4149418f750f59568ffa9feb2c45ba3479016554f53"

DROP = '\t{"WARP_STACK_SIZE", rtLimitType_t::RT_LIMIT_TYPE_SIMT_WARP_STACK_SIZE},\n'
KEEP = '\t{"DVG_WARP_STACK_SIZE", rtLimitType_t::RT_LIMIT_TYPE_SIMT_DVG_WARP_STACK_SIZE},\n'

OLD_LOOKUP = """	auto it = LimitTypeMap.find(type_str);
	if (it == LimitTypeMap.end()) {
	  printf("Invalid limit type: %s.\\n", type_str);
	  return NULL;
	}"""

NEW_LOOKUP = """	// CANN 9.1/9.2 do not declare RT_LIMIT_TYPE_SIMT_WARP_STACK_SIZE, so this limit
	// cannot be expressed against the installed runtime. Keep it explicitly
	// unavailable rather than guessing an equivalent enumerator: a silent remap
	// would change device-limit semantics. Remove this branch when a CANN release
	// declares the symbol and restore the map entry in the same change.
	if (std::string(type_str) == "WARP_STACK_SIZE") {
	  PyErr_SetString(PyExc_NotImplementedError,
	                  "WARP_STACK_SIZE is unsupported: the installed CANN runtime does not "
	                  "declare RT_LIMIT_TYPE_SIMT_WARP_STACK_SIZE");
	  return NULL;
	}
	auto it = LimitTypeMap.find(type_str);
	if (it == LimitTypeMap.end()) {
	  PyErr_Format(PyExc_ValueError, "Invalid limit type: %s.", type_str);
	  return NULL;
	}"""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> int:
    source = ORIGINAL.read_text()
    require(hashlib.sha256(source.encode()).hexdigest() == ORIGINAL_SHA256,
            "original npu_utils.cpp does not match the installed wheel bytes")
    require(source.count(DROP) == 1, "expected exactly one broken WARP_STACK_SIZE map entry")
    require(source.count(KEEP) == 1, "expected the DVG mapping to be present and unique")
    require(source.count(OLD_LOOKUP) == 1, "expected exactly one limit-map lookup block")

    patched = source.replace(DROP, "", 1).replace(OLD_LOOKUP, NEW_LOOKUP, 1)
    limit_map_body = patched.split("static const std::unordered_map")[1].split("};")[0]
    require("RT_LIMIT_TYPE_SIMT_WARP_STACK_SIZE" not in limit_map_body,
            "broken enumerator must not remain inside the limit map")
    require(patched.count(KEEP) == 1, "DVG mapping must be preserved verbatim")
    PATCHED.write_text(patched)

    import difflib
    diff = "".join(difflib.unified_diff(
        source.splitlines(keepends=True), patched.splitlines(keepends=True),
        fromfile="a/triton/backends/ascend/npu_utils.cpp", tofile="b/triton/backends/ascend/npu_utils.cpp"))
    (HERE / "npu_utils-cann92.patch").write_text(diff)

    record = {
        "original_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "patched_sha256": hashlib.sha256(patched.encode()).hexdigest(),
        "original_bytes": len(source.encode()),
        "patched_bytes": len(patched.encode()),
        "dropped_entry": DROP.strip(),
        "preserved_entries": [line.strip() for line in patched.splitlines() if "LimitTypeMap" not in line and line.strip().startswith('{"')],
        "unsupported_feature_stays_unavailable": True,
        "substituted_enumerator": None,
        "rollback": "restore npu_utils.cpp.orig over the installed npu_utils.cpp; byte-identical to the wheel",
        "cann_headers_checked": {
            "cann-9.1.0": {"path": "/usr/local/Ascend/cann-9.1.0/aarch64-linux/pkg_inc/runtime/rt_external_base.h",
                            "sha256": "e4f372e3a3804d8061f57f12714c415f8c5ea9d34905f9c726ecdfc0ae9696c6"},
            "cann-9.2.0": {"path": "/usr/local/Ascend/cann-9.2.0/aarch64-linux/pkg_inc/runtime/rt_external_base.h",
                            "sha256": "0b89cd3b978decfc25950509f8bb162ab1c3212631ff8d44e4734df29e2cb499"},
        },
    }
    (HERE / "patch-record.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())