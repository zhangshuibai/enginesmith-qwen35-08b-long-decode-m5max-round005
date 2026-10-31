# qwen35-08b-long-decode-m5max-round005

A specialized inference engine for [mlx-community/Qwen3.5-0.8B-4bit](https://huggingface.co/mlx-community/Qwen3.5-0.8B-4bit) at revision `da28692b5f139cb0ec58a356b437486b7dac7462`, built with EngineSmith 0.3.0.dev0.

- Machine: Apple M5 Max GPU (Metal), 40 cores, 128 GiB, macOS 27.0.1
- Reference: mlx-lm 0.32.2
- Judged under evaluation version 1
- Source and evaluation data: https://github.com/zhangshuibai/enginesmith-qwen35-08b-long-decode-m5max-round005
- Search: 12 rounds by codex

## Speed against mlx-lm (held-out final evaluation)

Score 1.36x; prefill 0.73x, decode 1.36x (reference time / engine time, 257 tokens per request).

| prompt tokens | prefill | decode |
|---:|---:|---:|
| 32,768 | 0.67x | 1.44x |
| 65,536 | 0.72x | 1.37x |
| 131,072 | 0.76x | 1.33x |

Correctness: passed; 3 of 8481 judged steps mismatch (at most 848 allowed).

## Speed in absolute terms

Medians over the paired rounds of the held-out final evaluation, this engine and mlx-lm on the same machine.

| prompt tokens | time to first token, ms | prefill tok/s | decode tok/s |
|---:|---:|---:|---:|
| 32,768 | 2,575.9 (mlx-lm 1,727.7) | 12,721.0 (18,966.6) | 483.1 (336.5) |
| 65,536 | 6,793.9 (mlx-lm 4,934.9) | 9,646.2 (13,280.0) | 359.8 (261.8) |
| 131,072 | 20,081.8 (mlx-lm 15,324.6) | 6,526.9 (8,553.1) | 239.7 (180.4) |

## How the search went

- 12 coding sessions by codex codex-cli 0.159.0-alpha.12.1; the search took 574 min.
- First correct engine: round 001, 25 min after the start.
- First engine as fast as mlx-lm (score 1.0): round 001, 25 min after the start.
- Agent usage: 1,926k input and 595k output tokens, 203,156k cache reads, cost not reported (as the agent reported them).

| round | kind | outcome | ended, min | score | prefill | decode | tokens in / out | engine lines |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 001 | generate | accepted (first_passing_candidate) | 24.7 | 1.02x | 0.73x | 1.02x | 78k / 27k | 10,065 |
| 002 | optimize | accepted (score_improved) | 66.9 | 1.15x | 0.72x | 1.15x | 93k / 21k | 10,617 |
| 003 | optimize | accepted (score_improved) | 109.4 | 1.23x | 0.72x | 1.23x | 95k / 36k | 11,876 |
| 004 | optimize | accepted (score_improved) | 162.8 | 1.34x | 0.73x | 1.34x | 154k / 51k | 13,131 |
| 005 | optimize | accepted (score_improved) | 204.7 | 1.37x | 0.73x | 1.37x | 131k / 42k | 20,484 |
| 006 | optimize | rejected (insufficient_improvement) | 272.0 | 1.38x | 0.99x | 1.38x | 198k / 74k | 29,999 |
| 007 | optimize | rejected (insufficient_improvement) | 322.5 | 1.39x | 0.73x | 1.39x | 198k / 53k | 25,873 |
| 008 | optimize | rejected (insufficient_improvement) | 372.3 | 1.37x | 0.97x | 1.37x | 172k / 54k | 25,945 |
| 009 | optimize | rejected (insufficient_improvement) | 441.9 | 1.38x | 0.82x | 1.38x | 359k / 86k | 34,843 |
| 010 | optimize | rejected (insufficient_improvement) | 486.5 | 1.36x | 0.92x | 1.36x | 130k / 54k | 24,144 |
| 011 | optimize | rejected (insufficient_improvement) | 541.2 | 1.37x | 0.93x | 1.37x | 198k / 53k | 24,026 |
| 012 | optimize | rejected (insufficient_improvement) | 574.2 | 1.37x | 0.97x | 1.37x | 119k / 45k | 21,270 |

`history.csv` in https://github.com/zhangshuibai/enginesmith-qwen35-08b-long-decode-m5max-round005 has every round with every prompt length; `enginesmith.json` has the same and more.

## Task accuracy (greedy decoding)

| benchmark | mlx-lm | this engine |
|---|---:|---:|
| MATH-500 (100 items) | 28.0% | 28.0% |
| IFEval, prompt-level strict (100 items) | 46.0% | 44.0% |

## Use it

These are the author's measurements. Check them on your own Mac (this builds the engine from `source/` and compares it with your local mlx-lm):

```sh
hf download mlx-community/Qwen3.5-0.8B-4bit --revision da28692b5f139cb0ec58a356b437486b7dac7462 --local-dir models/Qwen3.5-0.8B-4bit
uv run enginesmith verify https://github.com/zhangshuibai/enginesmith-qwen35-08b-long-decode-m5max-round005 --model models/Qwen3.5-0.8B-4bit
```

Build it yourself: `cargo build --release` in `source/`, then run `target/release/engine --model MODEL_DIR`; it speaks the engine protocol of EngineSmith's `docs/protocol.md` (token ids in, greedy token ids out). Read the source before you run it: it is native code written by a coding agent. This public package contains source only; the measured local binary is not distributed because it embeds build-machine filesystem paths. Build from `source/` to obtain the executable.

`enginesmith.json` holds every setting and result, and the SHA-256 of each model file.

## Compatibility and interpretation

The runtime currently requires Apple GPU family 10; the reported measurements were made on an Apple M5 Max. Compatibility and performance on other devices have not been established.

The held-out decode score is **1.36185x** (+36.18%, context weights 1/2/3 for 32K/64K/128K). Prefill is slower than mlx-lm: time to first token is 2.576 / 6.794 / 20.082 seconds versus 1.728 / 4.935 / 15.325 seconds. Passing the token-tolerance gates is not a claim of bitwise-equivalent output.

Both quality benchmarks completed 100 paired items with no recorded execution errors, using greedy decoding and an 8192-token cap. MATH-500 is 28% for both engines; IFEval strict prompt-level accuracy is 44% for this engine and 46% for mlx-lm. The paired difference standard error is approximately 3.48 percentage points for each benchmark. MATH-500 reached the token cap on 39 engine / 36 reference items; IFEval on 51 / 45. These finite, capped samples do not establish identical quality. Full output streams were identical on 10/100 MATH-500 and 25/100 IFEval items.

Files under `source/tools/` are historical development experiments, not supported user interfaces; some old probes are not standalone build targets. The supported build is `cargo build --release` from `source/`, and the engine accepts the EngineSmith token-ID protocol, not raw chat text. See `docs/protocol.md`.

## Source and measurement provenance

This is the source snapshot of selected round 005, commit `d8d561154c2b491224f2635baff241c479d6f1e1`. The measured local executable has SHA-256 `a7fe196508f54de5e700fd3c77f8d8ff7a6d57b713de1cba581fb6a9cce51397`. No runtime source or kernel was changed for publication. The public export adds this documentation and the protocol; it omits the original executable and model weights. Publication did not rerun GPU tests.
