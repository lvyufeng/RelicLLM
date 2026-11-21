"""Validate actual greedy token/logit artifacts after a successful inference exit."""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--attempt",required=True)
    args=parser.parse_args()
    root=Path(args.attempt).resolve()
    summary=json.loads((root/"summary.json").read_bytes())
    if summary["returncode"]!=0 or not summary["cleanup"]["complete"]:
        raise RuntimeError("inference did not exit successfully with complete owned cleanup")
    payload=torch.load(root/"generated-logits.pt",map_location="cpu",weights_only=True)
    tokens=[int(value) for value in payload["tokens"]]
    logits=payload["logits"].float()
    if not tokens or logits.ndim!=2 or logits.shape[0]!=len(tokens):
        raise RuntimeError("no complete generated-token/logit sequence was recorded")
    if not bool(torch.isfinite(logits).all()):
        raise RuntimeError("non-finite model logits cannot be claimed as successful inference")
    greedy=logits.argmax(-1).tolist()
    if greedy!=tokens:raise RuntimeError("recorded tokens do not match the actual greedy logits")
    tokenizer=AutoTokenizer.from_pretrained("/data1/relic-v41-20261116-210/checkpoint",local_files_only=True)
    output={"generation_succeeded":True,"token_ids":tokens,"text":tokenizer.decode(tokens,skip_special_tokens=True),
            "prompt_tokens":payload["prompt_tokens"],"logits_shape":list(logits.shape),
            "all_logits_finite":True,"tokens_match_greedy_logits":True,
            "generation_step_count":len(tokens),"checkpoint_rewritten":False,
            "official_reference_token_parity_claimed":False,"decode_graphs":payload["decode_graphs"],
            "owned_cleanup_complete":True}
    with (root/"validated-generation.json").open("x") as f:json.dump(output,f,indent=2)
    print(json.dumps(output,indent=2),flush=True)
    return 0


if __name__=="__main__":raise SystemExit(main())
