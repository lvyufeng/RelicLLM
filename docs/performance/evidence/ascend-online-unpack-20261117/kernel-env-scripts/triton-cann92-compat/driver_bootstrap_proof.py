"""Prove the patched Triton host-driver bootstraps through its real code path.

Instantiating ``triton.backends.ascend.driver.NPUUtils()`` is exactly what the
failing test did: it reads npu_utils.cpp, keys a cache entry on its md5, calls
``_build_npu_ext`` (the helper that raised), compiles it with the real torch /
torch_npu includes and CANN libraries, then dlopens the result.

This performs a host compile and dlopen only: no tensor allocation, no kernel
launch, no device memory. It also exercises the corrected limit behaviour.
"""
import json
import traceback

out = {}
try:
    from triton.backends.ascend.driver import NPUUtils
    utils = NPUUtils()
    module = utils.npu_utils_mod
    out["built"] = True
    out["module"] = getattr(module, "__name__", str(module))
    out["module_file"] = getattr(module, "__file__", None)
    try:
        module.set_device_limit(0, "WARP_STACK_SIZE", 1)
        out["warp_stack_size"] = {"raises": None, "note": "unexpectedly accepted"}
    except NotImplementedError as error:
        out["warp_stack_size"] = {"raises": "NotImplementedError", "message": str(error)[:200]}
    except Exception as error:  # noqa: BLE001
        out["warp_stack_size"] = {"raises": type(error).__name__, "message": str(error)[:200]}
    try:
        module.set_device_limit(0, "NOT_A_LIMIT", 1)
        out["invalid_name"] = {"raises": None, "note": "unexpectedly accepted"}
    except ValueError as error:
        out["invalid_name"] = {"raises": "ValueError", "message": str(error)[:120]}
    except Exception as error:  # noqa: BLE001
        out["invalid_name"] = {"raises": type(error).__name__, "message": str(error)[:120]}
except Exception:  # noqa: BLE001
    out["error"] = traceback.format_exc()[-3000:]
print("RESULT" + json.dumps(out))
