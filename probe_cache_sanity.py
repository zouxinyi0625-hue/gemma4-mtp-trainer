#!/usr/bin/env python3
"""Minimal check: is CacheDataset returning DISTINCT per-sample data, or the same
thing every row? probe_weight_vs_forward gave t0=3723 and tgt_next=107 for ALL
300 rows (different prompts, different anchors) with 100% hit — impossible unless
last_hidden is not actually varying per sample, OR the argmax path is wrong.

For the first N rows this dumps, at each row's first valid anchor a:
  - input_ids[a], input_ids[a+1]        (are the token streams even different?)
  - |last_hidden[a]|, first 4 values    (is the hidden varying per row?)
  - raw argmax(lm_head(last_hidden[a]))         (t0 candidate, no softcap)
  - argmax(lm_head(last_hidden[a+1]))           (tgt_next candidate)
and a distinct-value tally across all scanned rows.

If input_ids differ but last_hidden[a] is identical across rows -> cache read is
broken (shared buffer / mmap aliasing) and EVERYTHING downstream (incl. the cache
used for training) is suspect. If last_hidden differs but t0 is still constant ->
the lm_head/argmax path in the probe is wrong.

USAGE:
  python probe_cache_sanity.py \
      --target /tmp/models/gemma4/text_only \
      --cache-dir "$AZURE_ML_INPUT_ukwdata/maiprofile/mtp_26b/cache" \
      --n 12
"""
from __future__ import annotations

import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--n", type=int, default=12)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM
    from gemma4_mtp.target_cache import CacheDataset
    from gemma4_mtp.training_step import locate_target_parts

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt = torch.bfloat16

    print("loading target lm_head ...", flush=True)
    target = AutoModelForCausalLM.from_pretrained(
        args.target, dtype=dt, trust_remote_code=True).to(dev).eval()
    _, lm_head, _, _ = locate_target_parts(target)

    ds = CacheDataset(args.cache_dir)
    print(f"cache: {len(ds):,} samples  hidden_size={ds.hidden_size}\n")

    t0_seen, tnext_seen, hnorm_seen, iid_seen = set(), set(), set(), set()
    for i in range(args.n):
        s = ds[i]
        T = s["input_ids"].shape[0]
        lm = s["loss_mask"].to(torch.bool)
        valid = lm[:-1] & lm[1:]
        pv = torch.nonzero(valid).flatten()
        if pv.numel() == 0:
            print(f"row={i}: no valid anchor"); continue
        a = int(pv[0].item())
        iid_a = int(s["input_ids"][a].item())
        iid_a1 = int(s["input_ids"][a + 1].item())
        h = s["last_hidden"][a].to(dev, dt)
        h1 = s["last_hidden"][a + 1].to(dev, dt)
        hn = round(h.float().norm().item(), 3)
        head4 = [round(x, 3) for x in h.float()[:4].tolist()]
        with torch.no_grad():
            t0 = int(lm_head(h.unsqueeze(0)).argmax(-1).item())
            tnext = int(lm_head(h1.unsqueeze(0)).argmax(-1).item())

        t0_seen.add(t0); tnext_seen.add(tnext); hnorm_seen.add(hn); iid_seen.add(iid_a)
        print(f"row={i} anchor={a} seq_len={T}  input_ids[a]={iid_a} [a+1]={iid_a1}  "
              f"|h[a]|={hn}  h[a][:4]={head4}  t0={t0}  tgt_next={tnext}")

    print("\n----- distinct values across scanned rows -----")
    print(f"  distinct input_ids[a] : {len(iid_seen)}  {sorted(iid_seen)[:10]}")
    print(f"  distinct |last_hidden[a]| : {len(hnorm_seen)}")
    print(f"  distinct t0       : {len(t0_seen)}  {sorted(t0_seen)[:10]}")
    print(f"  distinct tgt_next : {len(tnext_seen)}  {sorted(tnext_seen)[:10]}")
    print("\n----- verdict -----")
    if len(iid_seen) > 1 and len(hnorm_seen) == 1:
        print("  input_ids differ but last_hidden[a] is IDENTICAL across rows ->")
        print("  CACHE READ IS BROKEN (mmap aliasing / shared buffer). The cache")
        print("  the model TRAINED on is likely corrupt too. This is the root cause.")
    elif len(hnorm_seen) > 1 and len(t0_seen) == 1:
        print("  last_hidden differs but t0 constant -> lm_head/argmax path wrong.")
    elif len(hnorm_seen) > 1 and len(t0_seen) > 1:
        print("  hidden and t0 both vary -> cache is FINE; the 100%/constant result")
        print("  in probe_weight_vs_forward came from that probe's own bug (likely")
        print("  the anchor selection collapsing to the same token). Re-examine it.")
    else:
        print("  inconclusive; inspect the per-row dump above.")


if __name__ == "__main__":
    main()
