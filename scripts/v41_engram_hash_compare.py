"""Compare the port's Engram hasher against the reference's, token by token.

The hashers address 189 GiB of n-gram tables and appear nowhere in the checkpoint, so a mismatch
here would send every position to the wrong rows and shift the residual at the two Engram layers.
This probe builds the reference's args straight from the config (so it never imports `model.py`,
which pulls tilelang), loads the reference's `engram.py` by path rather than by putting the
checkpoint's data directory on `sys.path`, and diffs the two hashers over the oracle's raw prompt.

Run with the port's env:

    PYTHONPATH=<relicllm checkout> python scripts/v41_engram_hash_compare.py \
        --checkpoint <ckpt> --inference <ckpt>/inference

Do not add `-I`: it implies `-E`, which drops `PYTHONPATH`, and the run then imports
whichever tree the environment's editable install points at rather than the one named above.
A probe that reports numbers from a revision nobody chose reads exactly like one that works.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from types import SimpleNamespace


def _plain(value):
    """A config value in JSON-comparable form: the port's tuples against the reference's lists."""
    return list(value) if isinstance(value, (list, tuple)) else value


def load_reference_module(inference_dir: str, name: str):
    """Import one module from the checkpoint's `inference/` tree, by path.

    By path rather than by putting the checkpoint's directory on `sys.path`: a checkpoint is data,
    and a data directory that lands on the import path can shadow any module name it happens to
    contain -- and it would stay there for the rest of the process.
    """
    path = os.path.join(inference_dir, f"{name}.py")
    if not os.path.isfile(path):
        raise SystemExit(f"{path} does not exist; is {inference_dir} a V4.1 inference tree?")
    spec = importlib.util.spec_from_file_location(f"_v41_reference_{name}", path)
    if spec is None or spec.loader is None:  # pragma: no cover - a readable file always has one
        raise SystemExit(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--inference", required=True, help="the checkpoint's inference/ dir")
    args = parser.parse_args()

    import numpy as np
    import torch
    from transformers import AutoTokenizer

    from relicllm.models.deepseek_v4_1.config import load_config
    from relicllm.models.deepseek_v4_1.loader import build_hasher

    ids = [19905, 418, 270, 4031, 2562, 28, 22471, 45117, 16]

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    config = load_config(f"{args.checkpoint}/config.json").text
    _, port_hasher = build_hasher(config, args.checkpoint, tokenizer=tokenizer)
    print("port token_map_size:", port_hasher.token_map_size, "pad_id:", port_hasher.pad_id)
    port_arr = np.array(port_hasher.hash_ids(ids, 0, None), dtype=np.int64)  # [L, layer, col]
    print("port hash shape:", port_arr.shape)

    # ---- reference, without importing model.py (which imports tilelang) ----
    engram = load_reference_module(args.inference, "engram")

    with open(f"{args.inference}/config.json") as f:
        raw = json.load(f)

    # The port derives its geometry from the checkpoint's top-level config, the reference from this
    # second copy under `inference/`; comparing the hashers is only meaningful if both files describe
    # one model, so every `engram_*` key of the reference's copy has to agree with the port's.
    for key in sorted(k for k in raw if k.startswith("engram_")):
        port_value = getattr(config, key, None)
        if _plain(port_value) != _plain(raw[key]):
            raise SystemExit(
                f"{key} disagrees between the two configs: port={port_value!r} ref={raw[key]!r}"
            )

    ref_args = SimpleNamespace(
        engram_layer_ids=raw["engram_layer_ids"],
        engram_max_ngram_size=raw["engram_max_ngram_size"],
        engram_n_heads=raw["engram_n_heads"],
        engram_vocab_size=raw["engram_vocab_size"],
        engram_num_embeddings=raw["engram_num_embeddings"],
        engram_head_dim=raw["engram_head_dim"],
        engram_pad_id=raw["engram_pad_id"],
        engram_compressed_vocab_size=raw["engram_compressed_vocab_size"],
        max_batch_size=1,
        max_seq_len=64,
    )

    class _TokWrap:
        backend_tokenizer = tokenizer.backend_tokenizer

        def __len__(self):
            return len(tokenizer)

    ref_layout = engram.EngramLayout.from_args(ref_args)
    ref_state = engram.NgramHashState(ref_args, ref_layout, _TokWrap())
    print("ref token_map_size:", len(set(ref_state.token_map.tolist())))
    t = torch.tensor([ids], dtype=torch.long)
    ref_arr = ref_state(t, 0, None)[0].to(torch.int64).cpu().numpy()  # [L, layer, col]
    print("ref hash shape:", ref_arr.shape)

    if port_arr.shape != ref_arr.shape:
        print("SHAPE MISMATCH", port_arr.shape, ref_arr.shape)
        return 1
    diff = port_arr != ref_arr
    n = int(diff.sum())
    print("total hash ids:", ref_arr.size, "mismatches:", n)
    if n:
        pos, lay, col = np.where(diff)
        for p, l, c in list(zip(pos.tolist(), lay.tolist(), col.tolist()))[:40]:
            print(f"  pos={p} layer={l} col={c}: port={port_arr[p,l,c]} ref={ref_arr[p,l,c]}")
    else:
        print("ALL ENGRAM HASH IDS MATCH")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())