# `--device` names a platform, and the platform is checked

**Affects:** anyone passing `--device` to `relicllm serve`, setting `DEVICE=…`, or building
`EngineArgs(device=…)` — and anyone whose launch relied on the flag being accepted and then ignored.

## What changed

`--device` was accepted, validated, forwarded to every rank, and then read by nothing. Two
consequences followed, and both were the same defect seen from two sides:

- **`auto` never resolved.** It is the default, and the flag's own help text has said "`auto` asks
  the build" since it was introduced — but no code in the repository asked anything. The one
  `auto` → concrete collapse in the tree is local to the `xing4` adapter.
- **`RuntimeCapabilities.devices` was never consulted.** Every runtime declares the platforms it can
  run on, and nothing compared that declaration to the request. `--backend mimo --device cpu` was
  accepted by a runtime whose own declaration is `("cuda",)`.

`auto` now resolves against what the host can actually serve, and the resolved platform is checked
against the selected runtime's declaration before anything is constructed.

| | Before | Now |
|---|---|---|
| `--device auto` (the default) on a CUDA host | Accepted; meant nothing | Resolves to `cuda`; checked against the runtime |
| `--device auto` on a host with no accelerator | Accepted; meant nothing | Resolves to `cpu`; a runtime declaring no `cpu` is refused |
| `--device ascend` with `torch_npu` absent | Accepted; failed later inside torch | Unchanged — see "What did not change". The refusal that names `torch_npu` is the device plane's, and the flag does not reach it |
| `--backend mimo --device cpu` | Accepted; the adapter has no CPU branch | `UnsupportedFeatureError` naming `mimo`'s declared set |
| `--device cuda` on a host with no CUDA | Accepted | Accepted — see "What did not change" |

The refusal reads (here for an `auto` that resolved to `ascend`; an explicit `--device ascend` gives
the same sentence without the "which is what …" clause):

```
backend='mimo' declares devices=('cuda',), so it cannot run on 'ascend', which is what
--device auto resolved to on this host. Ask for one of cuda, or use a runtime that serves 'ascend'
```

## What to do

**A launch that named a platform its runtime does not serve** — this is the one that will fail. The
message names both sides, so the fix is whichever of the two you actually meant:

| Launch | Was | Now |
|---|---|---|
| `--backend mimo --device cpu` | Silently ran | Refused. `mimo` has no CPU path; drop `--device`, or use `--backend torch`, which declares `("cuda","cpu")` |
| `--backend cpp --device cpu` | Silently ran | Refused. `cpp` declares `("cuda","ascend")`; `--backend torch` is the host-platform runtime |
| `--backend xing4 --device ascend` | Silently ran | Refused. `xing4` is CUDA-only |

**A launch on a host with no visible accelerator** — `--device auto` now resolves to `cpu`, so a
CUDA-only runtime is refused where it previously started and failed somewhere deeper. That is the
point: the failure moves to the flag instead of into a model load.

If you are running a CPU-only test on a card-bearing host, **`CUDA_VISIBLE_DEVICES=""` is now a
platform change as well as a card change.** `auto` resolves from `torch.cuda.is_available()`, so
emptying the variable answers `cpu` exactly as a cardless host does, and a runtime that declares no
host platform is refused:

```
backend='xing4' declares devices=('cuda',), so it cannot run on 'cpu', which is what --device auto
resolved to on this host
```

That is the same refusal a cardless host produces, and it is reached the same way — which is worth
knowing before using the spelling on a runtime whose declaration has no `cpu` in it. Pair it with a
runtime that declares `cpu` (`torch`, `v41`), or pass `--device cuda` explicitly if a CUDA-declaring
runtime has to be selected without the cards actually being used.

**Nothing needs to change for a launch that was already correct.** Every runtime's declared set
matches what it actually does; the check is a statement of the existing behaviour, not a new
restriction on it.

## What did not change

**An explicit platform the host lacks is still accepted.** `--backend torch --device cuda` fails at
the point CUDA is first touched, as it did before, and not at the flag. Whether a *runtime* can serve
a platform is a statement about that runtime's declaration; whether a *build* can is a statement
about torch's registered device types. The two are checked where those things are known — which is
also why `--device cuda` on a cardless box is not the flag's business to refuse.

`--device ascend` is the case that makes the split concrete. It is refused at the flag only when the
selected runtime does not declare `ascend` (`torch`, `v41`, `mimo` and `xing4` all refuse it, naming
their declared sets). `cpp` is the only runtime that declares it, so `--backend cpp --device ascend`
**passes the flag** and proceeds — and on a build without `torch_npu` it then fails where the platform
is first used, which is inside a model rather than at the flag. That is the same shape as
`--device cuda` on a cardless box, and it is the right one: the message an operator actually wants —
*this device type is not registered, install `torch_npu`* — is the device plane's, and it is raised by
`canonical_device`/`bind_device` at the call that binds the card, not by a flag check that has no
business knowing which accelerator packages are installed.

**`--device-ids` is unchanged.** The card list and the platform remain two flags answering two
questions, as they have since the two were split apart.

**The `cpp` backend's declaration is unchanged.** It declares `("cuda","ascend")` and no `cpu`. The
native engine does have a host-MoE offload path, but that is *weights in host memory*, not a CPU
execution plane, and the declaration says where the run executes. If that reading is wrong, the
declaration is what should change — not the check.

**The environment bridge is unchanged.** `DEVICE=…` and `POCKETLLM_DEVICE_IDS=…` are still read and
still mean what they meant; `DEVICE=ascend` now reaches the same check a flag would.
