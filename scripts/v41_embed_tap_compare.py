"""Compare a port stage tap against the reference's, as vectors, stage by stage.

The other tap comparisons in this tree answer narrower questions. The per-layer head projections
print absolute differences, which a growing stream norm turns into noise; a relative-L2 ladder
cannot tell a rotation from a scaling; and any of them that reconstructs a layer's head projection
through the *last* block's `hc_pre` is printing the forward's arithmetic for exactly one layer.

This asks the question that localises a defect in a 40-block stack: *which block first moves the
stream away from the reference's, and does the embedding agree before it.* It compares the two stages
both sides produce from the same place -- the embedding output, and each block's returned stream --
with cosine similarity, the norm ratio and the relative L2. The first block whose cosine drops is the
block that introduced the error; the embedding's cosine is the floor beneath it.

    python scripts/v41_embed_tap_compare.py <port-embed-tap.pt> <ref-embed-tap.pt>

The port file comes from `scripts/v41_embed_tap.py`, the reference's from `scripts/ref_embed_tap.py`
run against the checkpoint's own `inference/model.py`. Both record `h[i]` as the tensor the block
returned, which is the one quantity neither the norm, the head, nor a neighbour's `hc_pre` touches.

Reads `.pt` files on the box that wrote them; there is no torch on the development host.
"""
from __future__ import annotations

import sys

import torch


def load(path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict):
        raise SystemExit(f"{path}: expected a dict, got {type(obj).__name__}")
    return obj


def cos(a, b):
    a, b = a.reshape(-1), b.reshape(-1)
    return float(torch.dot(a, b) / (a.norm() * b.norm()).clamp_min(1e-30))


def pick(tensor, last_position=True):
    """One position of a captured stage. `[b, s, hc, d]` -> `[hc*d]` at the last token."""
    t = tensor.detach().float()
    if t.dim() == 4:
        t = t[0, -1] if last_position else t[0, 0]
    elif t.dim() == 3:
        t = t[0, -1] if last_position else t[0, 0]
    elif t.dim() == 2:
        t = t[0]
    return t.reshape(-1)


def line(name, a, b):
    a = a.detach().float().reshape(-1)
    b = b.detach().float().reshape(-1)
    na, nb = float(a.norm()), float(b.norm())
    rel = float((a - b).norm() / max(nb, 1e-30))
    print(f"  {name:<12} ||p|| {na:11.5f}  ||r|| {nb:11.5f}  ratio {na / nb:8.5f}  "
          f"cos {cos(a, b):9.6f}  relL2 {rel:.5e}")


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__.splitlines()[4].strip())
        return 2
    port = load(sys.argv[1])
    ref = load(sys.argv[2])

    print("embedding (the floor: no learned arithmetic between the ids and it)")
    if "embed" in port and "embed" in ref:
        line("embed", pick(port["embed"]), pick(ref["embed"]))
    else:
        print("  one side has no `embed`; skipping")

    print("\nblock streams, in order (h[i] is what Block[i].forward returned)")
    keys = sorted(set(port.get("h", {})) & set(ref.get("h", {})))
    first_bad = None
    for i in keys:
        a, b = pick(port["h"][i]), pick(ref["h"][i])
        c = cos(a, b)
        if first_bad is None and c < 0.9995:
            first_bad = i
        line(f"block {i}", a, b)
    print()
    if first_bad is None:
        print("every block's stream is within cos 0.9995 of the reference's")
    else:
        print(f"FIRST BLOCK BELOW cos 0.9995: {first_bad}")

    print("\npre_mix coefficients (the Hyper-Connections mix each block hands to the next)")
    pkeys = sorted(set(port.get("pre_mix", {})) & set(ref.get("pre_mix", {})))
    for i in pkeys[:8]:
        line(f"pre_mix {i}", pick(port["pre_mix"][i]), pick(ref["pre_mix"][i]))

    for name in ("attn", "ffn"):
        a_keys = sorted(set(port.get(name, {})) & set(ref.get(name, {})))
        if not a_keys:
            continue
        print(f"\n{name} sublayer outputs (blocks that recorded them)")
        for i in a_keys:
            line(f"{name} {i}", pick(port[name][i]), pick(ref[name][i]))

    print()
    print(f"port step0 argmax {port.get('step0_argmax')}  ref step0 argmax {ref.get('step0_argmax')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())