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
    rkeys = sorted(set(port.get("route", {})) & set(ref.get("route", {})))
    if rkeys:
        print("routing (the discrete decision: which experts each position picked)")
        print("  `set` is the question the MoE output depends on; `order` is only the ids' sequence,")
        print("  which differs whenever two experts are close and is not a different function.")
        first_flip = None
        for i in rkeys:
            a = port["route"][i].reshape(-1, port["route"][i].shape[-1])
            b = ref["route"][i].reshape(-1, ref["route"][i].shape[-1])
            if a.shape != b.shape:
                print(f"  layer {i:<4} shapes differ: {tuple(a.shape)} vs {tuple(b.shape)}")
                continue
            rows_a, rows_b = a.tolist(), b.tolist()
            same = sum(1 for ra, rb in zip(rows_a, rows_b) if ra == rb)
            same_set = sum(1 for ra, rb in zip(rows_a, rows_b) if sorted(ra) == sorted(rb))
            row = f"  layer {i:<4} set {same_set}/{len(rows_a)}  order {same}/{len(rows_a)}"
            if first_flip is None and same_set != len(rows_a):
                first_flip = i
                # which position: the first row whose expert *set* differs, and how its scores sit
                for j, (ra, rb) in enumerate(zip(rows_a, rows_b)):
                    if sorted(ra) != sorted(rb):
                        # `route_w` is recorded beside the ids, and a router that keeps a trailing
                        # axis records `[positions, k, 1]`. Flatten to `[positions, k]` rather than
                        # to the id tensor's shape, which would leave that axis in and hand the
                        # formatter a list of lists.
                        wa = port["route_w"][i].detach().float().reshape(-1, a.shape[-1])[j].tolist()
                        wb = ref["route_w"][i].detach().float().reshape(-1, b.shape[-1])[j].tolist()
                        only_a = sorted(set(ra) - set(rb))
                        only_b = sorted(set(rb) - set(ra))
                        row += (f"; first at position {j}\n"
                                f"      port ids {ra}\n"
                                f"      ref  ids {rb}\n"
                                f"      port-only {only_a}   ref-only {only_b}\n"
                                f"      port w   {[round(v, 5) for v in wa]}\n"
                                f"      ref  w   {[round(v, 5) for v in wb]}")
                        break
            print(row)
        print(f"\nFIRST LAYER WHOSE EXPERT SET DIFFERS: {first_flip}"
              if first_flip is not None else "\nevery layer routed to the same expert sets")
        print("A flip is not by itself a defect: it is what a small upstream difference does when two")
        print("experts are within that difference of each other at the top-k boundary. What it does")
        print("mean is that every layer below it is comparing two different functions, so a")
        print("cosine ladder that decays without a flip is a different finding from one that steps.")
        print()

    pt, rt = port.get("topk"), ref.get("topk")
    if pt and rt:
        print("sparse_attn gathered positions (one entry per call, in layer order)")
        if len(pt) != len(rt):
            print(f"  call counts differ: port {len(pt)}, ref {len(rt)}")
        for i in range(min(len(pt), len(rt))):
            a, b = pt[i].reshape(-1), rt[i].reshape(-1)
            if a.numel() != b.numel():
                print(f"  call {i:<3} widths differ: {a.numel()} vs {b.numel()}")
                continue
            same = int((a == b).sum())
            tail = "" if same == a.numel() else f"   first difference at {int((a != b).nonzero()[0])}"
            print(f"  call {i:<3} identical ids {same}/{a.numel()}{tail}")
        print()

    print(f"port step0 argmax {port.get('step0_argmax')}  ref step0 argmax {ref.get('step0_argmax')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())