# Redaction map

This file exists so that the claim "no internal addresses, no internal hosts, no credentials" can be **audited**
rather than trusted. It lists what the published material says instead of the real values, and what was scanned
before publishing.

## 1. Conventions used

| Instead of | The material says | Where |
| --- | --- | --- |
| node hostnames | "the head" / "the worker" / "the pair" | throughout |
| management and fabric addresses, subnets, HCA names | nothing at all — the interconnect is described as "one QSFP link between the CX7 ports, RoCE v2" with `<head-address>` as a placeholder | Requirements, Quick start |
| the public endpoint, domain, gateway, tunnel stack | nothing — the API is `http://<head-address>:8300/v1`, the engine's own default port | Quick start, Performance |
| weight and cache paths | "a prepared per-rank layout", "local NVMe" | Requirements, Window and slots |
| service names, tunnel ports, gateway configuration | nothing — no such name or port appears | throughout |
| credentials of any kind | nothing — no key, token, password, digest or certificate appears | throughout |
| the wider machine inventory | nothing — this note describes exactly one pair of GB10 systems | throughout |

Values that are **published deliberately**, because they are product behaviour or upstream defaults rather than
infrastructure: `CONTEXT`, `PARALLEL`, `KV_DTYPE`, `MAX_TOKENS`, `TF_DSV41_POOL_TOKENS`, the memory floors, the
`TF_DSV41_*` prefill and decode switches, the local API port `8300` (the recipe's own default), all throughput,
prefill and TTFT numbers, boot times, and the upstream error strings. The checkpoint is identified by its public
Hugging Face name and by its size class, because that is what a reader needs to reproduce the result; nothing about
where it is stored on our machines is published.

## 2. Categories scanned, with results

Scanned: every file in the published tree (`README.md`, `README.zh-CN.md`, `REDACTION-MAP.md`, `CHANGELOG.md`,
`CREDITS.md`, `NOTICE.md`, `LICENSE`, `tools/bench.py`).

| Category | Hits |
| --- | ---: |
| private IPv4 (RFC1918) | **0** |
| public IPv4 of any kind | **0** |
| our domains, subdomains or gateway hostnames | **0** |
| our hostnames, machine names, container names | **0** |
| enumerated service ports (tunnel, SSH, gateway) | **0** |
| credential patterns (`ghp_`, `hf_`, `sk-`, `AKIA`, `Bearer <value>`) | **0** |
| `password` / `passwd` / `secret` / `api_key` / `token` assignments | **0** |
| internal operational product names (tunnel, proxy, gateway stack) | **0** |
| private filesystem paths (`/srv/...`, `/home/...`) | **0** |
| deployment hostnames in commands (all are `<head-address>` / `<worker>` placeholders) | **0** |
| email addresses | **0** |
| MAC addresses, PCI addresses, serial numbers | **0** |

## 3. Informational hits, triaged and kept

* `127.0.0.1:8000` in `tools/bench.py` — a generic localhost default for a benchmarking script
  (`BASE` / `MODEL` are environment variables). Not our deployment.
* The local API port `8300` in the README — the upstream recipe's own default port, and the port a reader needs in
  order to follow the Quick start.
* Upstream project names, author handles and Hugging Face repository names in `CREDITS.md` and `NOTICE.md` —
  public attribution, which is required by the licenses of the work being built on.
* The words `password`, `secret`, `token` and `key` appear **only** inside this file, naming the categories that
  were scanned; no assignment or value of any of them appears anywhere in the repository.

## 4. Re-scan line

> Published tree: **residual sensitive hits = 0**, across the twelve categories above.
> Re-run this scan before any later update — the fields most likely to reappear are IP octets, subnets,
> enumerated ports, and any path copied out of a working configuration file.
