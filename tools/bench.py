#!/usr/bin/env python3
"""Client-side benchmark for an OpenAI-compatible TensorFold server.

Measures what a real client sees, over HTTP:
  * single-stream decode rate (first token -> last token), code / prose / structured
  * N-way same-workload decode aggregate (the cell that answers "does the pool serialise?")
  * a genuine long request with a needle planted mid-document (prefill rate + retrieval)

Usage:
    BASE=http://127.0.0.1:8000 MODEL=my-model python3 tools/bench.py [--streams 3]

Nothing here is deployment-specific: point BASE/MODEL at your server.
"""
import argparse, concurrent.futures as cf, json, os, time, urllib.error, urllib.request

BASE = os.environ.get("BASE", "http://127.0.0.1:8000").rstrip("/")
MODEL = os.environ.get("MODEL", "DeepSeek-V4.1-Flash-TF")
THINK_OFF = {"chat_template_kwargs": {"enable_thinking": False}}

# Workload texts: short programmatic prompts, the same shape the upstream engine benchmark uses.
CODE = ("Write a Python class `LRUCache` with `get(key)` and `put(key, value)` in O(1) "
        "using a dict and a doubly linked list. Include the class docstring and type hints. "
        "Return only the code.")
PROSE = ("Write a 400-word essay on why lighthouses were built where they were, how their keepers "
         "lived, and what replaced them. Plain prose, no lists or headings.")
STRUCTURED = "Count from 1 to 200, separated by commas, nothing else."
FILLER = ("The maintenance log for sector {i} records nominal readings, a routine inspection, "
          "and no outstanding items for the crew to follow up on this cycle. ")
NEEDLE = "\nThe access code for vault seven is ZEBRA-4417-QQ.\n"


def call(prompt, max_tokens=384, timeout=1800, extra=None):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0.0}
    body.update(THINK_OFF)
    if extra:
        body.update(extra)
    req = urllib.request.Request(BASE + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return {"err": e.read().decode()[:300]}
    except Exception as e:                                    # noqa: BLE001
        return {"err": f"{type(e).__name__}: {e}"}
    el = time.time() - t0
    usage = d.get("usage", {})
    ct = usage.get("completion_tokens", 0)
    return {"tok": ct, "pt": usage.get("prompt_tokens", 0), "wall": el,
            "tps": ct / el if el else 0, "t0": t0, "t1": time.time(),
            "content": (d["choices"][0]["message"].get("content") or "")}


def single_stream():
    print("== single stream, 384-token reply, greedy, thinking off ==")
    for name, prompt in (("code", CODE), ("prose", PROSE), ("structured", STRUCTURED)):
        for label in ("T=0", "T=0.7"):
            r = call(prompt)
            if "err" in r:
                print(f"  [{name:10s}] {label:5s} ERR {r['err'][:80]}")
            else:
                print(f"  [{name:10s}] {label:5s} {r['tok']} tok / {r['wall']:.2f}s "
                      f"= {r['tps']:.1f} tok/s")


def aggregate(prompts, max_tokens=128, tag="C3"):
    with cf.ThreadPoolExecutor(len(prompts)) as ex:
        res = [f.result() for f in [ex.submit(call, p, max_tokens) for p in prompts]]
    ok = [r for r in res if "err" not in r]
    if not ok:
        print(f"  {tag}: all failed")
        return
    first = min(r["t0"] for r in ok)
    last = max(r["t1"] for r in ok)
    total = sum(r["tok"] for r in ok)
    per = ", ".join(f"{r['tok']}/{r['wall']:.1f}s" for r in ok)
    print(f"  {tag}: {total} tok / decode window {last - first:.2f}s = "
          f"{total / (last - first):.1f} tok/s  (per slot {per})")


def concurrency(streams):
    print(f"\n== {streams}-way decode aggregate ==")
    aggregate([CODE, PROSE, STRUCTURED][:streams], tag=f"C{streams} mixed")
    # Same workload on every slot: this is the cell that shows pool serialisation, if any.
    aggregate([CODE + f"\n\n(variant {i})" for i in range(streams)], tag=f"C{streams}-same code")


def long_context(target_tokens):
    print(f"\n== long request, needle mid-document, target ~{target_tokens} tokens ==")
    parts, acc, i = [], 0.0, 0
    while acc < target_tokens:
        s = FILLER.format(i=i)
        parts.append(s)
        acc += len(s) / 4.3                      # rough chars-per-token ratio
        i += 1
    parts.insert(len(parts) // 2, NEEDLE)
    doc = "".join(parts) + "\n\nQuestion: what is the access code for vault seven? " \
                          "Answer with the code only."
    r = call(doc, max_tokens=40)
    if "err" in r:
        print("  ERR", r["err"][:200])
        return
    hit = "ZEBRA-4417-QQ" in r["content"]
    print(f"  prompt_tokens={r['pt']} | wall {r['wall']:.1f}s | "
          f"prefill {r['pt'] / r['wall']:.0f} tok/s | needle {'HIT' if hit else 'MISS'} | "
          f"content={r['content'][:40]!r}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--streams", type=int, default=3)
    ap.add_argument("--long", type=int, default=400000, help="target prompt tokens for the needle case")
    ap.add_argument("--skip-single", action="store_true")
    ap.add_argument("--skip-long", action="store_true")
    a = ap.parse_args()
    print(f"BASE={BASE}  MODEL={MODEL}")
    if not a.skip_single:
        single_stream()
    concurrency(a.streams)
    if not a.skip_long:
        long_context(a.long)
