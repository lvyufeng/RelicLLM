from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict

import torch

from src.gguf.ds4_mapping import validate_ds4_tensor_mappings
from src.gguf.reader import GGUFArraySummary, GGUFReader
from src.runtime.transformer import ModelArgs, Transformer


DS4_Q2_TYPES = {
    "q2_k",
    "iq2_xxs",
    "q4_k",
    "q8_0",
    "f16",
    "f32",
    "bf16",
    "i32",
}


def _format_bytes(value: int | None) -> str:
    if value is None:
        return "unknown"
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    amount = float(value)
    unit = units[0]
    for unit in units:
        if amount < 1024.0 or unit == units[-1]:
            break
        amount /= 1024.0
    if unit == "B":
        return f"{int(amount)} {unit}"
    return f"{amount:.2f} {unit}"


def _metadata_value_text(value) -> str:
    if isinstance(value, GGUFArraySummary):
        return f"array<{value.value_type_name}>[{value.length}]"
    return repr(value)


def _classify_tensor(name: str) -> str:
    if "ffn_gate_exps" in name or "ffn_up_exps" in name or "ffn_down_exps" in name:
        return "routed_experts"
    if "ffn_gate_shexp" in name or "ffn_up_shexp" in name or "ffn_down_shexp" in name:
        return "shared_experts"
    if ".attn" in name or "attn_" in name:
        return "attention"
    if "token_embd" in name or name.startswith("output") or ".output" in name:
        return "embedding_output"
    if "ffn" in name:
        return "ffn_other"
    return "other"


def _summarize(ds4):
    print(f"path: {ds4.path}")
    print(f"size: {_format_bytes(ds4.size)}")
    print(f"version: {ds4.version}")
    print(f"tensor_count: {ds4.tensor_count}")
    print(f"metadata_count: {ds4.metadata_count}")
    print(f"alignment: {ds4.alignment}")
    print(f"data_start: {ds4.data_start}")
    for key in (
        "general.architecture",
        "deepseek4.block_count",
        "deepseek4.embedding_length",
        "deepseek4.expert_count",
        "deepseek4.expert_used_count",
        "deepseek4.expert_feed_forward_length",
        "deepseek4.context_length",
    ):
        if key in ds4.metadata:
            print(f"metadata.{key}: {_metadata_value_text(ds4.metadata[key])}")

    by_type = Counter(t.type_name for t in ds4.tensors)
    by_class = Counter(_classify_tensor(t.name) for t in ds4.tensors)
    bytes_by_type = defaultdict(int)
    bytes_by_class = defaultdict(int)
    for tensor in ds4.tensors:
        if tensor.nbytes is not None:
            bytes_by_type[tensor.type_name] += tensor.nbytes
            bytes_by_class[_classify_tensor(tensor.name)] += tensor.nbytes

    print("\ntensor types:")
    for type_name, count in sorted(by_type.items()):
        print(f"  {type_name:12s} {count:6d} {_format_bytes(bytes_by_type.get(type_name, 0))}")

    print("\ntensor classes:")
    for class_name, count in sorted(by_class.items()):
        print(f"  {class_name:18s} {count:6d} {_format_bytes(bytes_by_class.get(class_name, 0))}")

    routed = [t for t in ds4.tensors if _classify_tensor(t.name) == "routed_experts"]
    routed_types = Counter(t.type_name for t in routed)
    print("\nrouted expert types:")
    for type_name, count in sorted(routed_types.items()):
        print(f"  {type_name:12s} {count:6d}")

    unknown_types = sorted({t.type_name for t in ds4.tensors if t.type_name.startswith("unknown_")})
    if unknown_types:
        print("\nunknown tensor types:")
        for type_name in unknown_types:
            print(f"  {type_name}")


def _print_tensors(ds4, limit: int, contains: str | None) -> None:
    tensors = ds4.tensors
    if contains:
        tensors = [t for t in tensors if contains in t.name]
    for tensor in tensors[:limit]:
        print(
            f"{tensor.name}\t{tensor.type_name}\t{tensor.dimensions}\t"
            f"offset={tensor.offset}\tabs={tensor.absolute_offset}\tbytes={_format_bytes(tensor.nbytes)}"
        )
    if len(tensors) > limit:
        print(f"... {len(tensors) - limit} more tensors")


def _validate_ds4_q2(ds4) -> int:
    errors: list[str] = []
    arch = ds4.metadata.get("general.architecture")
    if arch != "deepseek4":
        errors.append(f"general.architecture expected 'deepseek4', got {arch!r}")
    if ds4.version != 3:
        errors.append(f"GGUF version expected 3, got {ds4.version}")

    names = ds4.tensors_by_name
    required_metadata = (
        "deepseek4.block_count",
        "deepseek4.embedding_length",
        "deepseek4.expert_count",
        "deepseek4.expert_used_count",
        "deepseek4.expert_feed_forward_length",
    )
    for key in required_metadata:
        if key not in ds4.metadata:
            errors.append(f"missing metadata {key}")

    try:
        n_layers = int(ds4.metadata.get("deepseek4.block_count", 43))
    except Exception:
        n_layers = 43
    for layer in range(n_layers):
        for suffix, expected_type in (
            ("ffn_gate_exps.weight", "iq2_xxs"),
            ("ffn_up_exps.weight", "iq2_xxs"),
            ("ffn_down_exps.weight", "q2_k"),
            ("ffn_gate_shexp.weight", "q8_0"),
            ("ffn_up_shexp.weight", "q8_0"),
            ("ffn_down_shexp.weight", "q8_0"),
        ):
            name = f"blk.{layer}.{suffix}"
            tensor = names.get(name)
            if tensor is None:
                errors.append(f"missing tensor {name}")
            elif tensor.type_name != expected_type:
                errors.append(f"{name} expected {expected_type}, got {tensor.type_name}")

    unsupported = sorted({t.type_name for t in ds4.tensors if t.type_name not in DS4_Q2_TYPES})
    if unsupported:
        errors.append("unsupported tensor types: " + ", ".join(unsupported))

    if errors:
        print("validation: FAILED")
        for error in errors[:50]:
            print(f"  - {error}")
        if len(errors) > 50:
            print(f"  ... {len(errors) - 50} more errors")
        return 1
    print("validation: OK")
    return 0


def _runtime_state_shapes(config_path: str, routed_experts_device: str) -> dict[str, tuple[int, ...]]:
    with open(config_path) as f:
        config_data = json.load(f)
    config_data["routed_experts_device"] = routed_experts_device
    config_data["max_batch_size"] = 1
    config_data["max_seq_len"] = 16
    if "nextn_predict_layers" in config_data:
        config_data["n_mtp_layers"] = int(config_data.pop("nextn_predict_layers"))
    if "n_mtp_layers" not in config_data:
        config_data["n_mtp_layers"] = 0
    with torch.device("meta"):
        model = Transformer(ModelArgs(**config_data))
    return {key: tuple(value.shape) for key, value in model.state_dict().items()}


def _validate_runtime_mapping(ds4, config_path: str, routed_experts_device: str) -> int:
    validation = validate_ds4_tensor_mappings(
        ds4,
        _runtime_state_shapes(config_path, routed_experts_device),
    )
    print("\nruntime mapping:")
    print(f"  mappings: {len(validation.mappings)}")
    print(f"  missing_sources: {len(validation.missing_sources)}")
    print(f"  missing_targets: {len(validation.missing_targets)}")
    print(f"  shape_errors: {len(validation.shape_errors)}")
    print(f"  unmapped_sources: {len(validation.unmapped_sources)}")
    print(f"  unmapped_targets: {len(validation.unmapped_targets)}")
    errors = (
        validation.missing_sources
        + validation.missing_targets
        + validation.shape_errors
        + [f"unmapped GGUF tensor {name}" for name in validation.unmapped_sources]
        + [f"unmapped runtime tensor {name}" for name in validation.unmapped_targets]
    )
    if errors:
        print("runtime mapping: FAILED")
        for error in errors[:80]:
            print(f"  - {error}")
        if len(errors) > 80:
            print(f"  ... {len(errors) - 80} more errors")
        return 1
    print("runtime mapping: OK")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect a GGUF file without loading tensor payloads.")
    parser.add_argument("--gguf-path", required=True)
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--list-tensors", action="store_true")
    parser.add_argument("--contains", default=None, help="Only list tensors containing this substring")
    parser.add_argument("--limit", type=int, default=80)
    parser.add_argument("--validate-ds4-q2", action="store_true")
    parser.add_argument("--validate-runtime-mapping", action="store_true")
    parser.add_argument("--config", default="configs/config_w8a8.json")
    parser.add_argument("--routed-experts-device", choices=["gpu", "cpu"], default="cpu")
    args = parser.parse_args()

    if not os.path.exists(args.gguf_path):
        raise FileNotFoundError(args.gguf_path)
    ds4 = GGUFReader(args.gguf_path).read()
    if args.summary or not args.list_tensors:
        _summarize(ds4)
    if args.list_tensors:
        _print_tensors(ds4, args.limit, args.contains)
    status = 0
    if args.validate_ds4_q2:
        status = max(status, _validate_ds4_q2(ds4))
    if args.validate_runtime_mapping:
        status = max(status, _validate_runtime_mapping(ds4, args.config, args.routed_experts_device))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
