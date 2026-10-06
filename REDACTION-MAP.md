# REDACTION-MAP — how this repository keeps internal facts out

This file exists so the claim *"no internal addresses, no internal hosts, no credentials"* can be
**audited** rather than trusted. It lists what the published material says instead, the categories that
were scanned, and any residual hits with their triage.

## 1. Conventions used

| Instead of | The material says | Where |
|---|---|---|
| node hostnames | `rank0` / `rank1`, "the head", "the pair" | README §1-§6 |
| management/fabric addresses, subnets, HCA names | nothing at all — the fabric is described only as "one CX7 QSFP link, RoCE v2, 200 Gb/s" | README header table, §1.1 |
| our domain / gateway / tunnel stack | nothing — the API is described as "an OpenAI-compatible server on the local port" | README §5, §6 |
| weight paths | "a prepared per-rank layout" / "the local NVMe" | README §2.4, §5 |
| the gateway product, tunnel ports, frp/litellm config | nothing — no service name or port appears | throughout |
| credentials of any kind | nothing — no key, token, password or digest appears | throughout |
| the fleet's wider inventory | nothing — this repository describes exactly one pair of GB10 systems | throughout |

Values that are deliberately **published** because they are product behaviour, not infrastructure:
`CONTEXT`, `PARALLEL`, `TF_DSV41_POOL_TOKENS`, the memory floors, the `TF_DSV41_*` decode/prefill switches,
the throughput and prefill numbers, boot times, and the upstream error strings (they are the upstream
project's own messages and are what makes the failure modes reproducible).

The deployed checkpoint is identified by its **quantisation format and size class** (a 2.9 bpw EXL3 pack,
39 shards, ~197 GB) rather than by a repository revision, and the serving name used internally is not
published. Selection of a particular variant is not part of this deployment note.

## 2. Categories scanned, with results

Scanned: every file in the initial published tree (README, README.zh-CN, NOTICE, CREDITS, this file).

| Category | Hits |
|---|---:|
| private IPv4 (RFC1918: `10/8`, `172.16/12`, `192.168/16`) | **0** |
| public IPv4 of any kind | **0** |
| our domains, subdomains, or gateway hostnames | **0** |
| our hostnames, machine names, container names | **0** |
| enumerated service ports (frp, gateway, SSH, API) | **0** |
| credential patterns (`ghp_`, `hf_`, `sk-`, `AKIA`, `Bearer <value>`) | **0** |
| `password` / `passwd` / `secret` / `api_key` / `token` assignments | **0** |
| internal ops terms (gateway product names, tunnel/proxy stack names) | **0** |
| filesystem paths under a private root (`/srv/...`, `/srv2/...`) | **0** (paths are described, never quoted) |
| variant markers (`uncensored`, `abliterated`, `heretic`) | **0** |
| email addresses | **0** |
| MAC addresses | **0** |

Informational, expected hits that were triaged one by one and kept: the upstream project and author names
in §7 Credits (public attribution, required), the upstream error strings in §1.1-§1.2 (public project
output), and the HF organisation names in the credits (public model publishers).

## 3. Residual hits and triage

None. The initial tree scans clean across all twelve categories above.

## 4. Re-scan line

> Initial published tree: **residual sensitive hits = 0** across the twelve categories above.
> Re-run this scan before any later update — the fields most likely to reappear are IP octets, subnets,
> enumerated ports, and any path copied out of a working configuration file.
