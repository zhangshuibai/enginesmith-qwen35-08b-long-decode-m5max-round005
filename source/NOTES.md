# Current engine — session 005

Retained exact decode fusions: input RMS in grouped projections, combined Q/KV
preparation, and GDN input projection/convolution preparation. Long-context
launches fall from 202 to 154 per token. The GDN output fusion was rejected
because its isolated gain did not transfer to complete requests.

Production primitive audits, 45-case full-model parity in three request shapes,
95 protocol-state comparisons, and a byte-identical clean source build pass.
The combined engine measured **1.020628x weighted decode throughput versus the
session 004 parent (2.06% gain)** over six alternating pairs at each scored
length. All 18 streams matched; 17/18 decode pairs improved. The required
public two-round development evaluation passed, with weighted decode score
**1.364667x the reference**. All 18 controls passed; there were two allowed
rank-2 mismatches among 6,168 judged steps. The session 005 record below contains
mechanisms, complete measurements and rejected experiments.

# Previous best engine — session 004

Retained three bitwise-preserving changes: aligned vector K/V loads in
long-context attention, exact post-attention RMS inside decode MLP groups,
and exact logsumexp partials inside the vocabulary head. Decode dispatches
fall from 226 to 202; multirow prefill keeps its original path.

The complete candidate measured **1.075174x weighted decode speedup over the
session 003 parent (7.52% higher throughput)** in six alternating pairs at
32K/64K/128K. All 18 paired token streams were identical, and all 18 decode
pairs improved. Per-length paired gains were 4.78% / 6.21% / 9.33%.

The required two-round public `dev-eval` passed with weighted decode score
**1.339879x the reference**. All 18 controls passed with zero judged mismatches;
timed cases had two allowed rank-2 mismatches among 6,168 total judged steps.
This is development evidence, not the later five-round or held-out result.

Primitive parity, full-model comparisons, 78 protocol-state comparisons and
a byte-identical clean build passed. The session 004 record below includes
noise calibration, incremental and combined timing, rejected experiments,
packaging details and the public reference evaluation.

# Previous best engine — session 003

Retained four bitwise-preserving one-token fusions: GDN convolution/history/
preparation, grouped input projections, gate/up/SwiGLU, and QMV residual
addition. Decode dispatches fall from 424 to 226; multirow prefill is unchanged.

The complete candidate measured **1.065627x weighted decode speedup over the
session 002 parent (6.56% faster)** in six alternating pairs at
32K/64K/128K. Every paired token stream was identical. The two-round public
`dev-eval` passed with weighted decode score **1.237899x the reference**:
18 controls matched, with two allowed rank-2 mismatches among 6168 judged steps.
This does not claim the harness's later five-round or held-out result.

A clean source build was byte-identical; 117 protocol-state comparisons passed.
The session 003 log below records primitive parity, individual mechanism timings,
rejected attempts, noise, complete-candidate timings and reproducible-build data.

# Previous best engine — session 002

Retained changes: bounded GPU-ahead decode submission (first command buffer
one token, later buffers four, at most 16 steps outstanding), plus long-context
attention reduction/output-gate fusion with identical BF16 arithmetic.

The final all-length **two-round public development evaluation passed**:
weighted decode score **1.14511x the reference**. All 18 controls matched;
there were two rank-2 mismatches across 6168 judged steps, both the first
output at 32768. The 65536/131072 cases had no mismatches. This is development
evidence; the harness's later five-round/held-out acceptance is not claimed.

Parent-versus-scheduler paired timing showed **1.11147x task-weighted decode
speedup** over six rounds. The attention fusion separately added **1.01033x at
32768** over twelve rounds. These separately measured ratios are not multiplied
into a claimed combined parent comparison. Token parity, protocol-state checks,
clean builds, and rejected experiments are recorded below in the session 002 log.

# Qwen3.5-0.8B first passing engine — session 001

## Result

The first all-length development evaluation **passed**. Command:
`uv run enginesmith dev-eval --no-build --lengths all --rounds 1 --cooldown 15`.
Measured result is in session artifact `brief/dev-eval/001.json` (161 seconds).
All 18 control cases, including 8192 tokens, had zero judged mismatches.
The three scored cases had one rank-2 mismatch total, at the first output of
32768 tokens; 65536 and 131072 had zero mismatches. Each case generated 257
outputs. This is a one-round public development result, not the harness's
five-round final or held-out result.

| Prompt length | Prefill ref / engine (ms) | Decode ref / engine (ms, 256 outputs) | Decode speedup |
|---:|---:|---:|---:|
| 32768 | 1662.34 / 2656.18 | 753.17 / 800.14 | 0.9413 |
| 65536 | 4611.20 / 6621.78 | 967.49 / 952.60 | 1.0156 |
| 131072 | 14453.16 / 20379.93 | 1406.72 / 1449.01 | 0.9708 |

Weighted decode score: **0.98047**. Prefill geometric mean: **0.69035**.
This establishes a correct baseline; no speed improvement over the reference
is claimed. Hardware streaming bandwidth and compute floors were not measured.

## Final-build confirmation

The final source build passed the same all-length development command again:
`brief/dev-eval/002.json`, 135 seconds, one round with 15-second cooldowns.
All 18 controls again had zero judged mismatches; scored cases again had one
rank-2 mismatch at 32768 and zero at 65536/131072.

| Prompt length | Prefill ref / engine (ms) | Decode ref / engine (ms, 256 outputs) | Decode speedup |
|---:|---:|---:|---:|
| 32768 | 1657.54 / 2523.54 | 749.19 / 752.35 | 0.9958 |
| 65536 | 4604.94 / 6499.44 | 966.37 / 951.39 | 1.0157 |
| 131072 | 14372.22 / 20360.68 | 1398.27 / 1370.87 | 1.0200 |

Final one-round decode score: **1.01451**; prefill mean **0.69832**.
The two development runs range from 0.98047 to 1.01451; these were
correctness confirmations, not a controlled A/B performance experiment.
No numerical route changed between them.

## Implementation

Standalone Rust host with embedded MSL, compiled at startup with fast math off.
Pinned objc2 0.6.4 / objc2-metal 0.3.2; no Python, MLX, external kernel files,
or inference runtime is loaded by the executable. Runtime dynamic linkage is
only macOS system libraries, with no LC_RPATH. The checked loader copies the
694 text tensors from the affine Q4 safetensors checkpoint into aligned shared
buffers; vision weights are not used.

The model implements all 24 layers, BF16 array boundaries, FP32 GatedDeltaNet
state, full BF16 KV caches, convolution history, partial RoPE, tied output
embedding and BF16 log-softmax greedy tie handling. Reset and continuation
retain the protocol's pending-token invariant. Prefill takes chunks of up to
2048 tokens and processes the final prompt token separately. The decoder uses
one serial command buffer per token and waits after each, a simple baseline
with an obvious future scheduling opportunity that has not been measured.

Production dispatch uses reference g17s QMV row thresholds (33/25/13 by shape),
NAX matrix multiplication for larger row batches, NAX prefill attention for
rows>1, single-pass decode attention below 1024 context and 64-block split
attention from 1024 onward. All active pipelines compile before hello.
The stdin reader detects disconnect and stops at the next completed GPU step.

## Verification

- Offline locked release build succeeds with no warnings. A fresh source copy
  under `checks/clean-engine` also built offline from an empty target directory
  in 9.52 seconds, using cached locked Cargo dependencies.
- Native GDN probes with the actual Rust compile settings matched MLX bitwise
  at all eight intermediate stages for rows 1,2,19,2048: convolution/history,
  normalized QKV, decay, beta, FP32 state, readout and gated norm. The 2048-row
  case checks 12582912 BF16 values per QKV stage, 262144 FP32 state values and
  4194304 BF16 output values.
- Native linear probes: QMV matched 144/144 small-batch outputs; NAX matched
  7340032/7340032 outputs at M2048,N3584,K1024. Embedding, RMS and SwiGLU each
  matched 2626560 tested values. Greedy selection passed nine vocabulary tests.
- Native attention probes matched Q/K/V preparation and decode/gate bitwise
  at contexts 1,31,1023,1024,4096,131072. Long NAX prefill was exact at
  (rows,context)=(1024,1024),(2047,4095).
- Independent random full-model tests: 48 outputs at lengths 1,2,3 had no
  mismatches; another 455 outputs at lengths 8,16,33,128,257,1025,2049 had one
  rank-2 mismatch. These checks used the reference teacher-forcing judge.
- State tests at lengths 1,31,1023,2049,4097 verified 32-token full/split request
  equivalence, reset isolation, state preservation after rejected requests,
  mid-state hello, IDs/counts/ranges, minimal environment, and clean EOF exit.

The release binary still contains compiler/Cargo diagnostic source paths.
There are no host-specific model paths in the source or runtime dependency on
those diagnostic paths. The original and fresh-copy release binaries were byte-identical, SHA-256
`a9f6f350a69ba4f5eed113d8955fb3f61605b0f043a8af84b77d5ce11388e746`.
This verifies two source-directory paths on this host, not different Cargo or
toolchain locations. Dependencies are pinned in Cargo.lock; they are not vendored.

Final clean-copy state verification repeated all five lengths successfully,
with the same token hashes as the earlier build. During an 8192-token
continuation, closing stdin after 50 ms produced exit code 0 within 22.63 ms
and no partial/error reply (`protocol_state_final.json`). All test processes
were stopped and the shared GPU lock was released.

## Numerical findings and additional inactive routes

Plain FP32 sigmoid followed by one BF16 cast differed on 1893/10000 points;
stepwise BF16 exp, denominator, reciprocal, mirror and output multiplication
matched every point and is used. The first native GDN compile exposed that
MLX custom kernels implicitly supply log1p; the exact MLX helper was copied,
then verified using the standalone native runtime.

Small-row reference matrix products may use split-K. The baseline uses NAX
and has rare BF16 differences (e.g. 5/32256 and 49/131072 in probe runs).
`linear_splitk.metal` provides a separately tested, **inactive** exact route:
0/366096 differences over six reference split-K shapes. Its ABI and selection
rule are in LINEAR_INTERFACE.md. It is not wired into the production host;
forcing that route with splits=1 differs on 712/7340032 values, so NAX must
remain when the reference uses it. No performance effect was measured.

Two more **inactive** attention routes are retained for future correctness
work. `attn_small` matches the multi-query causal vector path bitwise for
2,3,8-row probes. The short BF16 score/softmax/value fallback reduces differences
from 32141/67584 values (NAX, rows33) to 3/67584, but is not universally exact
(e.g. 1825/2095104 at rows1023). Production remains the already-passing NAX
route for all multi-row attention. No performance effect was measured.

These alternatives were not integrated after the baseline passed, to leave
this session's delivered arithmetic as tested. Kernel ABI, generators, probe
commands and MIT source attributions are in LINEAR_INTERFACE.md,
GDN_INTERFACE.md and ATTENTION_INTERFACE.md. Only development probes use MLX.

# Session 002: decode scheduling and attention experiments

## Initial mechanism and verification plan

The parent calls `Gpu::finish` (commit plus wait) for every token. Tokens already
reside in `output_ids`, so this wait is not required by a host data dependency.
Hypothesis: serial per-token submission exposes host encoding and GPU wakeup
latency; keeping a bounded queue of steps submitted can remove those gaps.
The proposed change keeps every kernel, expression, reduction and dispatch order;
it submits the first decode step immediately, then batches four steps per command
buffer. Prefill chunks retain their existing partition. All work is drained before
reply, reset, cancellation or failure. Expected decode improvement is unmeasured;
no numerical latency estimate is claimed for this engine. Falsifier: paired decode
timing does not improve outside the measured A/A spread, or any token/state differs.

A separate isolated attention experiment will test reusing K/V loads across two
query heads while keeping the 64 interleaved block reduction and BF16 partial
boundary. The existing inactive GQA variant changes arithmetic order and is not a
candidate. Any attention change requires byte-for-byte intermediate parity and
end-to-end timing before integration.

## Scheduler correctness and noise measurements

- Parent A/A: `checks/aa-parent.json`, six paired rounds at P=32768,
  one warmup, 256 continuation tokens. Decode median A/B 1.00310,
  range 1.00075–1.00845, sample standard deviation 0.278 percentage points.
  Absolute medians 752.5/750.9 ms; all tokens identical. This measures the
  local comparison noise at 32K, not at the longer contexts.
- `same.py --long`: `checks/same-scheduler.json`, 45/45 prompts identical
  between parent and scheduler, three request shapes, 65 tokens each,
  including boundaries through 32768 and random-token prompts; both exits 0.
- `checks/state_check.py`: `checks/state-scheduler.json`, 24 shape cases
  at prompt lengths 3/1023/2049, counts 1/2/4/5/16/17/32/256, plus exact
  1+256 continuation, ten rejected requests, reset and mid-state hello:
  all passed. EOF during an active 65536-token decode exited 0 in 22.6 ms
  for the parent and 88.7 ms for the scheduler. No output is read before
  every submitted command completes. The outstanding bound is 16 steps.

The production long-context decode dispatch count remains 430 per token
(source-derived: 18 GDN layers x 18, six attention layers x 17, four
embedding/final-head dispatches). A 256-token decode now uses 65 command
buffers instead of 256, with GPU dependency order unchanged.

## Attention probes: rejected load sharing, candidate reduction rewrite

`engine/examples/attention_audit.rs` and `.metal` are isolated diagnostics.
The stage-1 variants keep each head's exact position sequence `block + 64*i`,
score dot-product order, online softmax order and BF16 partial storage.
Measured bitwise comparisons at contexts 1024/1025/4095/4096/32768/65536/131072,
two seeds, found zero differences in 131072 BF16 partials, 512 FP32 sums,
512 FP32 maxima and 2048 BF16 outputs per stage-1 comparison. The SIMD-lane
stage-2 remap also matched every tested output. Logs:
`checks/attention-audit/check-v2.jsonl` and `bench.jsonl`.

Isolated GPU microbenchmark medians, microseconds per dispatch (two seeds,
four alternating-order rounds per seed; 16 repeats for stage 1, 256 for stage 2):

| Kernel | 32768 | 65536 | 131072 |
|---|---:|---:|---:|
| Parent stage 1, TK4 | 119.40 | 259.21 | 529.65 |
| Stage 1, TK1 | 183.58 | 370.94 | 742.52 |
| Stage 1, TK2 | 128.11 | 268.77 | 540.41 |
| Stage 1, TK8 | 418.82 | 840.71 | 1682.56 |
| Shared two heads, TK1 | 194.19 | 396.72 | 784.35 |
| Shared two heads, TK2 | 173.62 | 347.80 | 694.99 |
| Shared two heads, TK4 | 239.74 | 477.56 | 953.12 |
| Shared two heads, TK8 | 644.67 | 1291.86 | 2581.12 |
| Shared four heads, TK4 | 326.82 | 648.57 | 1291.07 |
| Parent stage 2 | 5.15 | 5.14 | 5.15 |
| Stage 2 with lanes assigned to blocks | 1.83 | 1.83 | 1.83 |

All stage-1 variants are rejected as slower. Load sharing reduced parallel
query-head work and did not help this implementation. The stage-2 remap
removes 16 threadgroup barriers, preserving pairwise block additions and
SIMD reduction order. Its estimated whole-token saving is about 19.9 us
(six layers x 3.32 us); this is an estimate from a microbenchmark, not a
measured end-to-end speedup. A fused stage-2/output-gate experiment is prepared,
keeping the intermediate BF16 cast and all sigmoid rounding boundaries.

## Scheduler paired result

Measured normal protocol A/B, `checks/ab-scheduler.json`: six alternating
paired rounds, one warmup per length, fresh nonce prompts and 256 continuation
tokens. All 18 paired token streams were identical; both processes exited 0.
AC power; `pmset -g therm` reported no thermal/performance warnings. The GPU
lock excluded simultaneous GPU work and no builds ran during this timing.

| Context | Parent / scheduler decode median ms | Median speedup | Per-pair range | Ratio sd (percentage points) | Wins |
|---|---:|---:|---:|---:|---:|
| 32768 | 768.2 / 659.0 | 1.16660 | 1.15476–1.17427 | 0.716 | 6/6 |
| 65536 | 974.7 / 865.9 | 1.12548 | 1.12149–1.12848 | 0.312 | 6/6 |
| 131072 | 1387.2 / 1277.7 | 1.08458 | 1.08245–1.08846 | 0.254 | 6/6 |

Task-weighted decode improvement over the parent: **1.11147x**. Paired
prefill ratios were 1.0003/1.0075/1.0023; numerical prefill work is unchanged.
The decode saving is about 109 ms per 256-token request (about 426 us/token,
derived from the measured segment medians). The scheduling change is retained.

## Fused attention reduction/gate candidate

The isolated fused kernel passed 112 checks (seven contexts x four seeds x
four gate scales, including explicit signed zero and extreme +/-88 and +/-100
gates), plus 24 checks during its microbenchmark. Zero BF16 differences against
parent stage 2 followed by the original gate. `fused-check.jsonl` and
`fused-bench.jsonl` under `checks/attention-audit/` retain the raw results.

Measured isolated parent pair / remap plus separate gate / fused pair medians
in microseconds: 6.342/3.185/1.994 at 32K, 6.342/3.200/1.991 at 64K,
6.379/3.192/1.993 at 128K. The fused kernel retains the BF16 attention
boundary and every sigmoid cast; SIMD lanes retain the original block-index
reduction order. It eliminates the separate gate dispatch only for single-token
attention at context >=1024. Short decode and all multi-row prefill keep their
parent paths. Long-decode dispatch count becomes 424 per token.

The expected saving of 26.2 us/token is a microbenchmark-derived estimate.
Integration will be retained only after full-model token parity and a planned
12-round normal-protocol comparison against the scheduler-only build at 32K:
at least 0.5% median improvement and ten of twelve winning pairs. This small
attention effect is measured separately from the much larger scheduling win.

## Fused attention end-to-end result and packaging

The integrated production kernel was separately compiled by the diagnostic
probe and matched 229376 BF16 outputs in 112 context/seed/gate cases exactly;
all 476 comparisons in `checks/attention-audit/production-check.jsonl` passed.
`checks/same-fused.json` compared scheduler-only versus fused engine on all
45 `same.py --long` cases and three request shapes: 45/45 identical, clean exits.

Normal-protocol timing `checks/ab-fused.json`: twelve paired rounds at 32768,
two warmups, 256 continuation outputs. Scheduler-only/fused median decode
648.5/642.2 ms; median paired speedup **1.01033x**, 11/12 wins, range
0.99974–1.03535, ratio sample standard deviation 1.017 percentage points.
All tokens identical. Two positive timing outliers increased that spread;
the median still exceeds the predeclared 0.5%/10-win criterion. The measured
saving (~6.3 ms/request, or 24.6 us/token from segment medians) agrees with the
26.2 us/token microbenchmark estimate. The fused change is retained.
Prefill paired ratio 1.0091 is reported for completeness, not claimed as an
algorithmic gain: all prompt-chunk work is unchanged and only the last-token
attention tail differs.

A fresh source copy under `checks/clean-engine` built offline/locked from an
empty target directory in 10.46 seconds without warnings. Its release executable
is byte-identical to the measured production build, SHA-256
`0808715857fe13a51c5ab404b47a499d9a8d94f35c4739ed95d3ab6cb6404bf3`.
Runtime linkage remains macOS system frameworks/libraries only; no new runtime
or dependency was added. `target/` stays ignored. Diagnostic examples are
standalone probes and are not dispatched by the production engine.

## Final reference development evaluation

Command: `enginesmith dev-eval --no-build --lengths all --rounds 2 --cooldown 15`.
Result: `brief/dev-eval/001.json`, **PASSED**, 237.48 seconds, public prompts.
All 18 controls (including 8192) matched exactly. Two mismatches among 6168
judged steps; both rank 2 at the first output after a 32768-token prompt,
margin 0.0625, one in each round. No mismatches at 65536 or 131072.

| Context | Reference / candidate decode ms, round 1 | Reference / candidate decode ms, round 2 | Decode speedup | Prefill speedup |
|---|---:|---:|---:|---:|
| 32768 | 753.27 / 638.85 | 753.64 / 638.37 | 1.17983 | 0.65232 |
| 65536 | 967.99 / 840.13 | 973.74 / 839.07 | 1.15635 | 0.70696 |
| 131072 | 1410.66 / 1254.55 | 1414.15 / 1253.26 | 1.12641 | 0.72762 |

Weighted decode score **1.1451106**; prefill mean **0.7076658**. Prefill remains
slower than the reference, as in the parent; this session targets the decode-only
objective. The direct paired parent test is the evidence for the scheduling
improvement, rather than a cross-run comparison of these reference scores.
No outstanding measurements are needed for the retained changes. All engine,
probe and development-evaluation processes started for this session exited;
the GPU lock was released. No model, reference or harness source was changed.

# Session 003: reducing decode dispatch cost

Starting from session 002 (formal weighted decode score 1.1478), preserve the
parent's arithmetic and prefill partition. Candidate mechanisms under test:
GDN convolution/history/preparation fusion; residual-add plus RMS fusion;
MLP gate/up/SwiGLU fusion; exact parallel decomposition of vocabulary greedy
selection; unchanged-arithmetic attention scheduling. Each is checked against
the original kernels before full-model parent comparisons and paired timing.
No numerical performance estimate is claimed before measurement.

Parent binary saved at session artifact `checks/bin/parent`. Local parent A/A
at 32768 (six alternating rounds, one warmup, 256 continuation tokens) produced
median decode A/B 0.9981; median absolute times 640.9/643.3 ms, all paired
tokens identical (`checks/aa-parent.json`). Two short CPU probe builds overlapped
this preliminary noise measurement, so its spread is only provisional; final
comparisons will exclude builds. No GPU work overlapped it.

## Isolated checks and screening

- `gdn_conv_prepare_decode`: one SIMD group per Q/K/V head (48 groups) owns
  convolution channels and all their history. It keeps the four FMA taps,
  BF16 activation boundaries, four-element statistics walk, SIMD reduction,
  normalization and gate arithmetic. Replaces three decode dispatches by one,
  saving 36 dispatches/token. `checks/gdn-fusion/check.jsonl`: 320 evolving
  history/state cases, zero differing elements in history, normalized QKV,
  FP32 decay and state, beta, recurrence readout and gated output. Another
  80 checks passed before timing. Isolated prefix medians 4.40584/1.73250 us
  (parent/fused), 80/80 paired wins. Estimated saving 48.1 us/token; this
  estimate is not a measured full-model gain.
- `qmv_gate_up_swiglu_q4`: the same QMV lane walks for gate and up live in
  four SIMD groups; BF16 outputs pass through threadgroup memory into the
  original SwiGLU. Removes 48 dispatches/token. Isolated actual-weight
  medians 8.383/5.879 us for the parent three-dispatch chain/fused kernel;
  estimated saving 60.1 us/token. Actual 24-layer and synthetic parity checks
  reported zero differing BF16 values (`checks/linear-fusion/`).
- `add_rms_norm_bf16`: residual add retains its BF16 cast before the unchanged
  two-level RMS reduction. Synthetic shape/alias and actual-weight checks
  pass. Isolated parent add+RMS/fused medians 2.394/1.792 us. Under comparison
  with an alternative QMV residual epilogue; not yet retained.
- Rejected vocabulary split: moving the existing 32 SIMD-group reductions
  into separate groups and keeping their walks/reductions passed all 512
  synthetic vocabulary token checks, but was slower: parent median 43.48 us,
  split 44.91 us (12 alternating pairs, 128 repetitions each). Diagnostic
  code is only in `examples/greedy_audit.rs` and `greedy_split_003.metal`;
  it is not compiled or dispatched by production. `checks/greedy-audit.jsonl`.
- Rejected attention layout calibration: unchanged-arithmetic S2 groups gave
  isolated median paired ratios 1.01242/1.00519/1.02791 at 32K/64K/128K;
  range and spread were too noisy for the estimated sub-1.5% weighted
  full-model effect. The separate exp(0)/multiply-by-one branch experiment
  changed BF16 partials/outputs and is rejected for parity failure.
  `checks/attention-schedule-003/summary.json` records all 20 variants.

- `qmv_multi_q4`: independently projected matrices share one dispatch while
  retaining each original matrix's N, row/lane walk and reduction. Saves
  54 GDN and 12 attention launches/token. Synthetic 672 checks/1286880 BF16
  values and real 1080 checks/4290048 values had zero differences. Isolated
  per-layer medians: GDN 12.080/6.389 us, attention 8.751/4.820 us. A full
  24-layer streaming loop measured 366.482/230.811 us; 135.7 us/token saving
  is a microbenchmark estimate until whole-model timing confirms it.
- `qmv_add_q4`: chosen residual mechanism. Keep the projected BF16 value,
  add the existing residual, then BF16-store in place to the residual.
  Every output row has one owner; the QMV input is a distinct gated/activated
  buffer. Removes 48 add dispatches/token. Real 1152 checks/4718592 BF16
  values and synthetic 576 checks/842112 values, including alias cases,
  had no differences. Streaming all 48 matrices with following RMS gave
  371.835/334.510/330.938 us for parent/add+RMS/QMV+add; selected QMV+add
  for slightly better isolated time and simpler host dependencies.
  The unselected add+RMS kernel remains an inactive diagnostic alternative.
- Combined candidate decode dispatch count: 226 rather than 424, from the
  source map (36+48+66+48 launches removed). Multirow prefill routes and
  attention's numerical work are unchanged. Independent source review
  checked the buffer ABI, launch coverage, history ownership, barriers,
  residual aliases and exactly one residual update per stage.

The candidate's one-token map (source-derived):

| Stage | Dispatches |
|---|---:|
| Embedding | 1 |
| Each GDN layer: input norm, grouped projections, conv/preparation, recurrence, gated norm, output+residual, post norm, gate/up/SwiGLU, down+residual | 9 × 18 |
| Each attention layer: input norm, grouped projections, Q prepare, KV prepare, split attention, reduce/gate, output+residual, post norm, gate/up/SwiGLU, down+residual | 10 × 6 |
| Final norm, vocabulary QMV, greedy | 3 |
| Total | 226 |

## Incremental full-model verification and timing

GDN fusion: `checks/same-gdn.json`, 45/45 prompts, three request shapes each,
65 outputs, including random prompts and boundaries through 32768, identical
parent/candidate tokens and clean exits. `checks/ab-gdn.json`, six alternating
pairs at P32768, one warmup, 256 continuation outputs: decode median
642.123/626.583 ms, median paired speedup **1.024759x**, 6/6 wins, range
1.023886–1.042854, sample SD 0.750 percentage points. Prefill ratio 1.008660
is reported but not claimed as a prefill algorithmic improvement. All paired
tokens identical. AC power, no thermal/performance warning; builds and other
GPU jobs excluded during these comparisons. This retains the GDN mechanism.

MLP fusion on top of GDN: `checks/same-mlp.json`, 45/45 prompts and three
request shapes identical, clean exits. `checks/ab-mlp.json`, six alternating
pairs at P32768, one warmup, 256 continuation outputs: 645.633/631.500 ms,
median paired speedup **1.021654x**, 6/6 wins, range 1.018649–1.031222,
sample SD 0.473 percentage points. Prefill ratio 1.001371; no prefill gain
claimed. All paired tokens identical. Absolute timings drifted relative to
the earlier GDN experiment; only within-experiment ratios are used. This
retains the MLP mechanism.

Grouped projections on top of GDN+MLP: `checks/same-projections.json`,
45/45 prompts × three request shapes identical, clean exits.
`checks/ab-projections.json`, six alternating P32768 pairs, one warmup,
256 continuation outputs: median 626.039/591.985 ms, paired speedup
**1.050622x**, 6/6 wins, range 1.044083–1.067396, SD 0.825 percentage
points. Prefill ratio 1.001516; no prefill gain claimed. Tokens identical
in every pair. This retains shared-input grouped projections. Separate
incremental ratios are not multiplied into a combined improvement claim;
a direct complete-candidate versus original-parent comparison follows.

Residual QMV epilogue on top of the preceding three mechanisms:
`checks/same-residual.json`, 45/45 prompts × three request shapes identical,
clean exits. `checks/ab-residual.json`, six alternating P32768 pairs, one
warmup, 256 continuation outputs: median 605.723/594.675 ms, median paired
speedup **1.016625x**, 6/6 wins, range 1.007062–1.039853, SD 1.147 percentage
points. Prefill ratio 0.998734; all paired tokens identical. The positive
outlier increases spread, but all six pairs improve and the median exceeds
the original noise level. This retains the simpler QMV residual epilogue.
The alternative add+RMS remains unused by production.

## Final candidate packaging and protocol state

A fresh source copy in `checks/clean-engine` built offline/locked from an empty
`target/` in 9.26 seconds without warnings. It is byte-identical to the measured
candidate: SHA-256
`3a75f4acc5ae4db8ed9d87c3626491ce8db35f60cc6cbe7eaa21cdc699a8a1be`.
Runtime linkage is only macOS system libraries/frameworks. No runtime dependency
or model file was added, and `target/` remains ignored.

`checks/state_check_003.py` compared original parent and clean-build candidate:
117 streams matched, including monolithic/split counts 1/2/4/5/16/17/32/256
at P3/1023/2049, continuation chains, exact 1+256, reset after a longer prompt,
mid-state hello and 11 rejected requests followed by continued generation.
IDs, counts and token ranges were checked. EOF during a long decode after a
65536-token prompt exited cleanly in 88.67/88.69 ms (parent/candidate), with
no partial reply. Report: `checks/state-final.json`, 36.87 seconds elapsed.

## Final reference development evaluation

`enginesmith dev-eval --no-build --lengths all --rounds 2 --cooldown 15`
passed in 235.36 seconds: `brief/dev-eval/001.json`. All 18 controls,
including 8192, had zero mismatches. There were two rank-2 mismatches among
6168 judged steps: only the first output after P32768, once in each round,
margin 0.0625. P65536 and P131072 had no mismatches. This is a two-round
public development result, not the harness's later five-round/held-out test.

| Context | Reference/candidate decode ms, round 1 | Reference/candidate decode ms, round 2 | Decode speedup | Prefill speedup |
|---|---:|---:|---:|---:|
| 32768 | 749.886 / 565.924 | 753.861 / 566.414 | 1.328000 | 0.652786 |
| 65536 | 966.779 / 765.989 | 974.542 / 767.593 | 1.265869 | 0.706956 |
| 131072 | 1404.660 / 1182.270 | 1412.444 / 1182.328 | 1.191367 | 0.733707 |

Weighted decode score **1.23789945**, prefill mean **0.71070189**. Prefill is
still slower than the reference; no prefill algorithm change is claimed.
The direct paired parent comparison, rather than comparing this score with
another session's score, determines the combined optimization effect.

## Final timing noise calibration

`checks/aa-parent-final.json`: four alternating parent/parent pairs at every
scored context, one warmup per length, 256 continuation outputs, with no
concurrent builds or GPU jobs. All tokens identical, both exits 0. Measured
median decode A/B ratios: 0.997346 / 0.996957 / 1.001762 for 32K/64K/128K;
weighted ratio 0.999422. Sample SD: 1.391 / 1.270 / 0.469 percentage points.
Ranges: 0.995133–1.024279 / 0.977723–1.008145 / 0.993971–1.004530.
Thus isolated pairs occasionally drift by about 2–2.5%; the final combined
comparison uses six alternating pairs at every length. AC power; `pmset`
reported no thermal/performance warning, though absolute times did drift.
`checks/ab_progress.py` is the supplied AB tool with progress logging added
between completed pairs; request generation, timing and comparisons are unchanged.

## Complete candidate versus original parent

`checks/ab-final.json`: six alternating paired rounds, one warmup per scored
length, 256 continuation outputs, normal protocol, shared GPU lock. All 18
paired streams were identical and both processes exited 0. Measured task-weighted
decode improvement is **1.06562748x**. No per-mechanism ratios are multiplied.

| Context | Parent / candidate decode median ms | Median paired ratio | Pair range | SD, percentage points | Wins |
|---|---:|---:|---:|---:|---:|
| 32768 | 668.386 / 610.664 | 1.092850 | 1.077837–1.102066 | 0.821 | 6/6 |
| 65536 | 871.005 / 807.439 | 1.078736 | 1.067596–1.099207 | 1.204 | 6/6 |
| 131072 | 1279.267 / 1221.313 | 1.048127 | 1.042771–1.057821 | 0.587 | 6/6 |

Paired prefill ratios at 32K/64K/128K: 0.997542 / 0.999813 / 0.999067.
No prefill algorithmic gain is claimed. Every retained mechanism improved in
its own six-pair comparison, and the complete candidate improved at all
three scored lengths. Final tests ran with no concurrent builds or GPU probes.
All engine, probe and evaluation processes launched in this session exited;
the GPU lock was released. No model, reference or harness source was changed.

# Session 004: explicit attention cache loads

Retained mechanism: the existing long-decode attention stage 1 now loads each
lane's eight contiguous BF16 K/V values through one 16-byte uint4 expression,
then bitcasts into the original arrays. Cache layout, TK=4 / block+64*i walk,
launch geometry, tails, arithmetic, reductions and BF16 boundaries are unchanged.
The compile-time assertion restricts the template to eight 16-bit values/lane.
All addresses are 16-byte aligned; the existing main-loop guard covers all four
rows. Independent review checked alignment, bounds and unchanged ownership.
The observed mechanism is the explicit vector memory expression and timing gain;
GPU instruction disassembly was not collected.

Primitive verification: 112 cases, 14 contexts 1024..262144 including tails,
eight seeds each; zero differences in 14,680,064 BF16 partials, 57,344 FP32 sums,
57,344 FP32 maxima and 229,376 BF16 outputs. The retained audit is
`examples/attention_production_004.rs` with its renamed original source
`attention_decode_parent_004.metal`. Build with `cargo build --release --example
attention_production_004`, then run under the supplied GPU lock with arguments
`check 1024,1025,1030,1031,2047,4095,4096,32768,32769,65536,65537,131072,131073,262144 8`.
Session artifact `experiments/attention/check-production.jsonl` records all cases.

Isolated stage 1, 32 alternating pairs: parent/candidate medians (us)
125.816/111.215, 264.290/234.564, 537.488/469.318 at 32K/64K/128K;
median ratios 1.14642/1.13473/1.15373. The 32K probe was noisy (range 0.66–1.60),
so these are diagnostic timings only. The 128K estimate was 409 us/token saved
across six attention layers, not a measured request improvement.

Full-model `same.py --long`: 45/45 prompts, three request shapes, 65 outputs each,
identical to original parent; clean exits. `checks/same-attention.json`.
Parent A/A (`checks/aa-parent.json`), four alternating pairs at all scored
lengths: decode median ratios 0.995884/1.001177/0.997014; sample SD 0.795/0.582/
0.324 percentage points. Weighted ratio 0.998211; all tokens identical.

Normal-protocol parent/attention A/B (`checks/ab-attention.json`), six alternating
pairs per length, one warmup, 256 continuation tokens, shared GPU lock and no
concurrent builds/probes: all 18 paired streams identical, both exits 0.

| Context | Parent/candidate decode median ms | Median paired ratio | Range | SD, percentage points | Wins |
|---:|---:|---:|---:|---:|---:|
|32768|605.958/596.514|1.015833|1.00434–1.04073|1.461|6/6|
|65536|804.907/768.293|1.046680|1.04389–1.05338|0.382|6/6|
|131072|1217.892/1129.841|1.079214|1.07242–1.09411|0.801|6/6|

Measured task-weighted decode improvement: 1.05753668x (5.75%). Prefill ratios
0.998971 / 0.997368 / 1.000630; no prefill algorithm gain claimed. AC power;
`pmset` reported no thermal/performance warning. These paired measurements,
rather than comparing scores from different sessions, justify retention.

Rejected session 004 explorations (all isolated; no production routes added):

- Folded KV cache by 64 interleaved blocks: exact primitives, but no extra win
  after controlling launch grouping/vector loads. Winning normal-layout S4
  uint4 ratios 1.1338/1.1398/1.1697 versus folded 1.1191/1.1345/1.1635.
  Extra cache/layout complexity is not justified. Eight-byte loads were
  neutral: 0.9969/1.0009/1.0039. `experiments/attention/README.md`.
- GDN recurrence+norm in 512 threads/head: 320 evolving checks and 80 benchmark
  checks exact, but parent/fused 3.5082/3.7830 us, median ratio 0.927375,
  only 4/80 wins. Physical state transpose: 20 multirow cases through 2048 rows
  and 320 evolving cases exact; isolated ratio 0.976461 (7/80 wins), 18-state
  streaming 88.695/90.552 us, ratio 0.980410 (1/18 wins).
  `experiments/gdn/{README.md,evidence.json}`.
- Separate exact FP32 attention-score dispatch: 196 primitive cases exact
  over seven contexts/four seeds/seven variants. All variants slower.
  Best ratios 0.8334/0.8802/0.9210 at 32K/64K/128K, 16 pairs each;
  parent 118.428/256.771/540.746 us versus best 142.134/291.645/584.776 us.
  `checks/logs/attention-scores-{check,bench}.jsonl`.

The retained linear changes and final packaging/reference evaluation are
recorded below. The attention-only result above does not include them.

## Session 004: RMS prologue in the MLP

Retained `qmv_mlp_g8_norm` repeats the original 256-thread, eight-SIMD RMS
calculation within each gate/up threadgroup. Each group keeps its private
normalized BF16 input, then computes 16 output rows with the original QMV
walk and SwiGLU boundaries. This removes 24 post-attention norm dispatches
per decode token. Multirow prefill keeps its original path. Raw residual,
normalized output and activation buffers are distinct; only group zero
publishes the normalized vector. Independent source review found no alias,
bounds, reduction-order or barrier issue for M=1, K=1024 and 256 threads.

Primitive audit: the selected variant matched all 144 real-weight cases
(24 layers × six activation scales), including 516,096 activated BF16
values and 147,456 normalized BF16 values. The broader seven-variant audit
matched 1,008 cases. `examples/norm_mlp_audit_004.rs` directly compiles the
retained production source; arguments are `check MODEL_DIR` under the GPU
lock. The original experiment is `experiments/linear/norm-mlp-check.jsonl`.

Isolated parent norm+MLP / fused medians were 7.27669 / 6.40723 us over 12
alternating pairs. The paired ratio was 1.13455, with 12/12 wins, but large
positive timing outliers (range 1.12751–1.69832) made the spread wide.
The resulting 20.87 us/token saving was only an estimate before integration.
Groups 16 and 32 did not give a clear improvement and were not selected.

Full-model `checks/same-norm.json`: attention-only versus attention+norm,
45/45 prompts × three request shapes × 65 outputs identical; both exits 0.
`checks/ab-norm.json`: 12 alternating 32K pairs, two warmups, 256 continuation
outputs, no concurrent builds/probes. Parent/candidate median decode
550.8486 / 543.9029 ms; median paired speedup **1.0125224x**, 11/12 wins,
range 0.971927–1.014322, sample SD 1.179 percentage points. All tokens match.
One negative outlier increases the spread. The median and win count pass the
predeclared criterion (at least 1.005x and 10/12 wins). Prefill ratio 1.002961
is reported without claiming a prefill algorithm gain. The final combined
all-length comparison below measures the complete retained engine.

## Session 004: exact logsumexp partials in the vocabulary head

Retained `head_lse_g8` assigns one SIMD group to each of the original 1,024
threads in `greedy_bf16`. It computes that thread's original four logits per
4,096-row tile with the unchanged QMV walk, retains the BF16 logit boundary,
and performs its online max/normalizer update immediately. The separate tail
keeps the original 1,024-thread reductions, BF16 logsumexp and subtraction,
and lowest-token-ID tie handling. All vocabulary logits are still stored.
The last tile's inactive positions remain negative infinity exactly as before.

The head uses 128 groups of 256 threads. Two disjoint 4,096-byte slices of the
existing, now-dead attention temporary hold 1,024 FP32 maxima and normalizers.
The serial encoder orders reuse and the slices are in bounds. Independent
review checked ownership, aliases, tail positions and all reduction orders.
No new allocation, dependency or decode dispatch is introduced.

Primitive evidence for the selected variant: 48 cases (eight seeds × six
scales, including all-zero/all-tied logits), zero differences in 11,919,360
BF16 logits, 49,152 FP32 maxima, 49,152 FP32 normalizers and all 48 tokens.
The broader four-grouping audit passed 192 cases. The production source is
checked directly by `examples/head_lse_audit_004.rs`, with the original greedy
partial walk in `head_lse_reference_004.metal`. Arguments: `check MODEL_DIR`,
under the GPU lock. `checks/logs/head-production-final.jsonl` passed all 48.
The final norm and attention production audits also repeated successfully.

Isolated head+greedy parent/fused medians: 302.91309 / 284.07519 us, 12 pairs;
median ratio 1.06844 with 12/12 wins, but large positive timing outliers.
The resulting 18.84 us/token estimate was not an end-to-end measurement.
Groups 1/2/4 were slower (311.24/312.01/311.63 us) and were rejected.

Full-model `checks/same-head.json`: attention+norm versus all three changes,
45/45 prompts × three request shapes × 65 outputs identical, clean exits.
`checks/ab-head.json`: 12 alternating 32K pairs, two warmups, 256 continuation
outputs, no concurrent builds/probes. Median decode 544.2949 / 541.7395 ms;
paired speedup **1.0051031x**, 11/12 wins, range 0.969309–1.008007, sample
SD 1.050 percentage points. Every paired stream matches. One negative outlier
increases the spread. This small result narrowly passes the predeclared 1.005x,
10/12-win criterion; the microbenchmark overestimated its practical saving.
Prefill ratio 1.000666 is reported without claiming a prefill algorithm gain.
Individual mechanism ratios are not multiplied into a combined speedup claim.

Other linear screens were not retained. More SIMD groups per ordinary head
QMV were neutral: group-8 ratio 1.00415, 8/12 wins; large groups slowed small
matrices. Static K=1024 address strides gave head ratio 1.00538, 9/12 wins,
with wide outliers, and were not extended to production. These screens had
1,620 synthetic and 216 real-weight comparisons with exact outputs.
`experiments/linear/README.md` records all raw paths and statistics. Preloading
the head's input blocks across tiles was prepared but not built or measured;
no gain is claimed and it is not in production.

Source-derived final long-decode dispatch count: 202 instead of 226 (the
normalization fusion removes 24; vector loads and head/LSE keep the count).
The required final combined comparison and reference evaluation follow.

## Session 004: clean build and protocol state

A clean source copy in `checks/clean-engine` built offline/locked from an empty
target directory in 8.72 seconds without warnings. It exposed a stale root
incremental executable: copying host source with preserved old modification
times had prevented Cargo from rebuilding the main binary while examples
rebuilt successfully. The independently measured experiment binaries were
checked and contain their intended mechanisms. Touching the integration inputs
and rebuilding the root binary fixed the artifact; no source or numerical
change was needed. Final root and clean-copy executables are byte-identical:
SHA-256 `27a177b7dbbd8b11a93ea74e649f072e6f1e2160bf4a0b5739a23a7689624c48`.
All final request timing and reference evaluation use this verified executable.
Runtime linkage is only macOS system frameworks/libraries; no LC_RPATH, model
files, external shader files or runtime dependency was added. `target/` stays
ignored. Clean-copy equality is verified on this host, not other toolchains.

`checks/state_check_004.py` compared the original parent with that clean build.
All 78 paired stream comparisons passed, including monolithic/split counts
1/2/4/5/16/17/32/256 at P3/P1023/P2049, a mixed continuation chain, mid-state
hello, reset after a longer prompt, and continuation after 12 rejected requests.
IDs, counts and vocabulary ranges were checked. EOF during a long decode after
P65536 exited cleanly in 88.72/88.63 ms (parent/candidate), with no partial
reply. Both processes exited 0; total check elapsed 38.16 seconds. This verifies
protocol-visible state behavior, not a direct full-model state-byte dump.

## Session 004: complete candidate versus the session 003 parent

`checks/ab-final.json` measures the verified final executable against the
unchanged session 003 parent: six alternating pairs at each scored length,
one warmup per length, 256 continuation outputs, normal JSON-lines protocol,
shared GPU lock, and no concurrent builds or GPU probes. All 18 paired token
streams were identical; both processes exited 0. Times below are separate
arm medians, while speedups are medians of paired ratios.

| Prompt length | Parent / final decode (ms) | Paired speedup | Ratio range | Sample SD (percentage points) | Wins |
|---:|---:|---:|---:|---:|---:|
| 32768 | 588.004 / 566.368 | 1.047822 | 1.019930–1.052427 | 1.430 | 6/6 |
| 65536 | 786.078 / 740.089 | 1.062143 | 1.057681–1.073492 | 0.582 | 6/6 |
| 131072 | 1196.345 / 1096.305 | 1.093301 | 1.081390–1.096068 | 0.554 | 6/6 |

Task-weighted geometric mean with weights 1/2/3: **1.075174x decode speedup
over the parent (7.52% higher decode throughput)**. This directly measures
all three retained mechanisms together; individual screen ratios are not
multiplied. Every paired decode ratio exceeds one despite gradual timing
drift during the run. The earlier A/A ratios and dispersion above provide
the noise comparison.

Prefill parent/final medians were 2734.952 / 2736.044 ms, 7062.098 / 7037.521
ms and 20704.416 / 20638.660 ms at 32K/64K/128K. Paired prefill ratios were
0.999605 / 1.002822 / 1.001925; no prefill algorithm improvement is claimed.


## Session 004: required public reference evaluation and handoff

Final command: `enginesmith dev-eval --no-build --lengths all --rounds 2
--cooldown 15`. `brief/dev-eval/001.json` reports **PASSED**, complete, no
failure, in 235.07 seconds. All 18 controls passed with zero judged mismatches.
Sixteen control streams were identical; control-p2047 and control-p2048 differed
as streams but had zero mismatches under the reference's forced-token judging.
There were two allowed rank-2 mismatches across 6,168 total judged steps, both
the first output at 32768 (one per round). The 65536/131072 timed cases had
no mismatches. Each case generated 257 outputs (one prefill, 256 continuation).

Times below are separate arm medians; speedups are medians of paired ratios.

| Prompt length | Prefill reference / final (ms) | Decode reference / final (ms) | Paired decode speedup | Mismatches by round |
|---:|---:|---:|---:|---:|
| 32768 | 1672.884 / 2533.916 | 755.898 / 540.024 | 1.399748 | [1, 1] |
| 65536 | 4650.017 / 6533.265 | 973.392 / 719.901 | 1.352120 | [0, 0] |
| 131072 | 14544.350 / 19987.631 | 1413.521 / 1076.957 | 1.312515 | [0, 0] |

Task-weighted decode score: **1.339878620x the reference**. Weighted prefill
ratio: **0.711086111x**. Prefill has zero task weight and remains slower than
the reference. This two-round public development result does not establish
the harness's later five-round acceptance, held-out result or quality scores.
The controlled parent comparison above establishes this session's improvement;
scores from separate sessions are not used as an A/B experiment.

No model, reference, harness, protocol, dependency or runtime-launch code was
changed. Production changes and their primitive audits are committed; target
artifacts remain ignored. All engines, probes and benchmark processes started
by this session exited. The shared GPU lock was confirmed available and
released. Source changes ended before final timing; only this record changed
after the verified clean build.


# Session 005: exact decode fusions

## Parent and experiment selection

The offline/locked parent build exactly reproduced session 004's executable,
SHA-256 `27a177b7dbbd8b11a93ea74e649f072e6f1e2160bf4a0b5739a23a7689624c48`.
`checks/plan005.md` records mechanisms and falsifiers before production edits.
The parent has 202 dispatches per long-context decode token. Input RMS still
runs separately before every layer's grouped projection; GDN output gated norm
runs separately before its residual projection; Q and KV preparation run as
independent adjacent dispatches. These are the targeted boundaries.

A/A calibration (`checks/aa-parent.json`) ran three alternating pairs at each
scored length, one warmup, 256 continuation outputs. All streams were identical,
both processes exited 0, and no builds or GPU probes overlapped timing.
32K/64K/128K median decode ratios were 1.002721/1.005484/1.000727; sample SD
0.247/0.722/0.153 percentage points. The weighted ratio was 1.002643. This small
calibration exposes local drift; it is not a speedup. AC-power thermal checks
reported no warning. Later acceptance uses paired requests and repeated wins.

## Input RMS inside grouped projections

The selected `qmv_multi_g8_norm` uses the exact parent eight-SIMD RMS calculation
in each 256-thread group. All groups read the distinct raw residual; only group
zero publishes normalized scratch. Eight SIMD groups compute 32 projection rows
with the same four-row SIMD QMV expressions, K walk, reductions and BF16 boundary.
All barriers precede returns for the short 16-row projections. Multirow prefill
keeps its original norm and projections. This removes 24 decode dispatches.

Primitive selected-kernel checks: 576 cases (24 real layer weight sets × four
seeds × six scales), zero differences in 4,290,048 projection BF16 values and
589,824 normalized BF16 values; poisoned guards unchanged. The full grouping
screen passed 2,304 cases. Diagnostic streaming over all 24 layers measured
parent/fused 280.870/258.120 us, 16 alternating pairs, median ratio 1.087614,
range 1.07012–1.09346, 16/16 wins. The 22.75 us/token saving was an estimate
from this diagnostic, not a complete-request measurement. Plain group8 was
neutral (1.00071); norm group16/group32 ratios were 1.04222/1.05907 and were
not selected. Evidence: `checks/linear005/README.md` and raw JSONL.

## Shared dispatch for attention preparation

`attn_prepare_qkv_decode` maps the original eight Q and two KV threadgroups into
one 10-group launch. Its uniform branch copies the original arithmetic, exact
two-SIMD RMS reduction, BF16 boundaries, RoPE helpers and cache indexing. Outputs
are disjoint and no group consumes another group's writes. Only single-row
calls use it. This removes six decode dispatches.

Primitive production audit passed 240 cases: 12 positions from 0 through
262143 × four seeds × five scales, with query outputs, active/adjacent KV rows
and the final full cache bytes exact. The isolated 16-pair diagnostic measured
parent/fused medians 2.9354/1.5628 us, median ratio 1.8874 and 16/16 wins, with
large timing outliers (ratio range 1.516–4.890). The implied 8.24 us/token saving
was only an estimate. `checks/logs/prepare-{fusion,production}.jsonl` records
raw results. Whole-model retention is determined below.

## Initial combined correctness and source review

Direct production-source primitive audits passed for input RMS, the later-rejected
GDN output fusion and combined attention preparation.
`checks/same-initial.json`: 45/45 prompt cases, three request shapes and 65
outputs per shape, all tokens identical to the session 004 parent; both exits 0.
Independent reviews checked fixed-shape bounds, buffer aliases, ownership,
barriers, reduction orders and exact host bindings. No findings; see
`checks/attention005/review.md` and `checks/linear005/independent-review.md`.
No whole-model speedup is inferred from multiplying isolated gains.

## Incremental complete-request timing

`checks/ab-input.json`: parent versus input RMS fusion, 12 alternating pairs at
32K, two warmups, 256 continuation outputs. All tokens identical; both exits 0.
Decode medians 541.546/532.423 ms, median paired speedup 1.017007, 11/12 wins,
range 0.975622–1.018067, sample SD 1.200 percentage points. One negative outlier
widens the spread. Prefill ratio 0.999644. This clears the predeclared 1.005x
and 10/12 wins threshold and supports retaining input RMS fusion.

`checks/ab-gdn.json`: input RMS versus input RMS + GDN output fusion, same
12-pair 32K protocol. All tokens identical; both exits 0. Decode medians
556.358/560.564 ms, paired ratio 0.997158, only 5/12 wins, range 0.986899–1.004338,
sample SD 0.577 percentage points. Prefill ratio 1.002899. **Rejected**: the
isolated norm/output gain did not transfer to the full chain. The diagnostic did not reproduce the full decode chain; the later larger
working-set probe below also failed to predict the observed request result.
No specific counter-level cause was established. Production reverts
to the parent GDN output path. Its exact primitive results remain useful
negative evidence; no gain from that fusion is claimed.


`checks/ab-prepare.json`: input RMS + GDN output fusion versus the same plus
combined Q/KV preparation, 12 alternating 32K pairs, two warmups,256 continuation
outputs. All tokens identical; both exits 0. Decode medians 575.739/570.684ms,
median paired ratio 1.008624,12/12wins,range1.000626–1.022834, sample SD 0.784
percentage points. Prefill ratio 0.998902. This passes the predeclared threshold.
The rejected GDN output fusion was present in both arms of this incremental
measurement; the final combined parent comparison will separately establish
the result after removing it. The selected path has input RMS and combined
attention preparation, pending the additional GDN input experiment below.

## Exact GDN input projection and convolution fusion

The selected `gdn_input_g16` combines input RMS, QKV/Z/A/B projections,
convolution/history, Q/K normalization and recurrence gates in one decode
launch. It replaces the already-fused input RMS/projection launch plus the
convolution/preparation launch, removing 18 more dispatches per token.
Multirow prefill retains its original operations and launch order.

Each of 64 threadgroups has 512 threads. The first 48 groups own one complete
128-channel Q/K/V block each; the last 16 own Z blocks. Every group repeats
the parent's exact eight-SIMD input RMS. Sixteen SIMD groups compute two
original four-row QMV walks each and store the projected BF16 values in
threadgroup memory. After a uniform barrier, SIMD group zero performs the
original four-value-per-lane convolution and Q/K normalization. Each Q block's
second SIMD group computes the original four-row A/B projection group, selects
its head's BF16 row, and computes the original recurrence gates. History has
one owner per channel, and no group reads another group's projection/history.
All projected intermediates and normalized input are still published.

Primitive screen `checks/gdn005/input-bench.jsonl`: 1,296 exact comparisons
across all 18 real GDN layers, six activation scales, four evolving-history
steps and G8/G16/G32 variants. Zero differences in 43,856,640 BF16
intermediate/history values and 20,736 FP32 decay values; all 64-byte guards
remain unchanged. The G16 subset contains 432 comparisons, 14,618,880 BF16
values and 6,912 FP32 decay values. The selected production source is audited
directly by `examples/gdn_input_audit_005.rs` (`check MODEL_DIR`, shared GPU
lock); the production rerun passed all 432 cases in
`checks/logs/gdn-input-production.jsonl`.

The isolated 18-layer input-weight cycle is about 85 MB (estimate). Over 24 alternating pairs, parent RMS/projection plus
convolution/preparation measured 244.02279 us versus G16 223.33138 us per
18-layer cycle. Median paired ratio was 1.094631, 24/24 wins, range
1.079128–1.098114. G8 measured 1.047853 (23/24 wins), G32 1.040222 (24/24).
The 20.69140 us saving is a diagnostic estimate of token savings, not a
normal-protocol request result. Complete-model evidence: `checks/same-gdn-input.json` passed all 45 prompt cases,
three request shapes and 65 outputs per shape against the original parent.
`checks/ab-gdn-input.json` compares input RMS + combined preparation with and
without this extra fusion: 12 alternating 32K pairs, two warmups, 256 decode
outputs. Every stream is identical; both exits are 0. Median decode times
536.762/532.669 ms; paired ratio **1.005026**, 12/12 wins, range
1.000841–1.043632, sample SD 1.137 percentage points. Prefill ratio 1.006307.
One large positive outlier widens the spread. The result narrowly clears the
predeclared 1.005x / 10-of-12 threshold, so the fusion is retained. The final
combined all-length comparison below separately measures the net effect.

Independent review by the attention agent checked prototype ownership,
uniform barriers, bounds and expression preservation. A self-audit also
checked the production source against the selected measured body, all 26
host buffers, shape parameters, launch geometry, aliasing and preserved
multirow branch. The production audit/full-model checks provide numerical
verification; source similarity alone is not treated as proof.

Rejected GDN experiments:

- Gated RMS as the output QMV prologue: 432 real-weight checks exact across
  G2/G4/G8/G16. G8's hot 18-layer cycle appeared faster, 98.27604/83.60743 us,
  ratio 1.175477 with 24/24 wins, but this ~21 MB diagnostic did not reproduce complete decode requests.
  Normal-protocol input-RMS-only versus input-RMS+G8 comparison over 12
  alternating 32K pairs measured ratio **0.997158**, only 5/12 wins, range
  0.9869–1.0043; all tokens matched. The G8 output fusion was removed.
  The selected output primitive rerun also passed 108 cases with separate
  output and in-place residual/output aliasing (663,552 BF16 values checked).
- A follow-up output-projection screen used 90 independent copies of the
  real matrices (~106 MB, estimate), all checkpoint bytes unchanged. Parent
  measured 557.71875 us/90 layers; G2/G4/G8/G16 measured
  748.29036/575.58985/508.31641/484.04297 us. Paired ratios were
  0.745930/0.969711/1.097112/1.151678, wins 1/1/24/24 of 24. One large parent
  outlier inflated maxima. This diagnostic does not reverse G8's full-model
  rejection; G16 was not integrated or measured in the full engine.
- Explicit aligned float4 state loads/stores in the existing recurrence:
  2,880 evolving comparisons plus 360 pre-benchmark comparisons exact.
  At the production two-SIMD geometry, parent/candidate medians were
  3.51318/3.51611 us including the unchanged gated norm; ratio 1.001227,
  19/32 wins, neutral. One-SIMD scalar/vector variants measured roughly
  3.43164/3.42399 us, a tiny ~0.082 us absolute difference from the parent
  with wide outliers. Larger groups were neutral/slower. Neither vector
  state I/O nor recurrence launch tuning is retained.

Raw experiment logs and the full rationale are in `checks/gdn005/README.md`.
All probes ran under the shared GPU lock, with no concurrent builds/GPU jobs,
and exited cleanly. No model, harness, reference or dependency was changed.


## Other rejected screens

Head input-block preload across vocabulary tiles, aligned packed-weight loads,
and their combination were bitwise exact in 144 cases (eight seeds × six
scales × three variants), including all BF16 logits, FP32 LSE partials and
selected tokens. Diagnostic 16-pair head ratios were 0.99193 / 1.00095 /
0.98658. None justified integration; the parent head remains unchanged.
`checks/linear005/README.md` records the raw timing and parity results.

Rejected session 005 attention screens: delayed V loads/early score computation,
compact uint4 live arrays, and SIMD-shared exponential evaluation all preserve
primitive bits but give no useful speedup. The exp variants evaluate the original
two/eight arguments across SIMD lanes then broadcast, retaining every original
multiply/add; they do not simplify exp(0) or multiply-by-one as the previously
failed numerical experiment did. Delayed loads/packed live state were tested
on session 004's vector-load baseline, a changed condition from scalar session003.

`examples/attention_lifetime005.{rs,metal}` and
`checks/attention005/{README.md,check.jsonl,screen-summary.json}` record the
experiment. 420 comparisons at seven contexts (1024 through 131072, including
tails), six query scales/seeds and ten variants: zero differences in 55,050,240
BF16 partials, 215,040 FP32 sums, 215,040 FP32 maxima, 860,160 BF16 outputs.
The broad forward/reverse isolated screen used 12 samples per variant/length;
first 32K samples have large timing outliers. TK4 lifetime/packed variants
ranged from 0.99673 to 1.00362x; paired-exp ratios were
1.00280/1.00430/1.00458x at 32K/64K/128K, batched-exp
1.00365/1.00347/0.99946x. TK2 was 8–13% slower, TK8 4–7% slower.
All fail the predeclared 1.02x isolated screen threshold and remain inactive.
No full-model performance or token-parity claim is made for these variants.

A separate exact prefill-grid-order screen was also rejected. It remapped only
logical threadgroup query/head indices before the unchanged NAX body, testing
all-heads-fast, GQA-heads-fast, alternating KV heads and reversed query tiles.
165 comparisons across 11 shapes (including tails and 128K), three scales and
five variants including zero-remap control matched all BF16 outputs.
`examples/prefill_schedule_005.{rs,metal}` and `checks/prefill005/README.md`
retain the audit. Twelve broad forward/reverse isolated samples per shape gave
best head-fast ratios 1.00252/1.00821/1.00656 at 32K/64K/128K; all long-chunk
variants failed the 2% screen criterion amid material outliers. Reversed query
tiles gave 1.14196x on the first 2K chunk but only~0.425ms estimated savings
across its six layers; later long chunks were neutral. No full-model gain is
claimed and the production prefill grid remains unchanged.


## Packaging and final validation

Source-derived final long-decode dispatch count: **154**, versus 202 in the
session 004 parent. Input RMS fusion removes 24; combined Q/KV preparation
removes six; the extra GDN input/convolution fusion removes 18. Multirow
prefill arithmetic and partitioning are unchanged. All runtime changes are
Rust host bindings and embedded MSL; no dependency, model, reference, harness
or protocol changes were made.

All examples build offline/locked without warnings. A clean source checkout
under `checks/clean-engine` built offline/locked from an empty target directory
in 9.05 seconds. Root, measured candidate and clean-checkout executable bytes
are identical, SHA-256
`a7fe196508f54de5e700fd3c77f8d8ff7a6d57b713de1cba581fb6a9cce51397`.
This verifies reproducibility across source directories on this host, not
across toolchains. Runtime linkage is only macOS system frameworks/libraries.
`target/` remains ignored. Rejected shaders are isolated development examples
and are not compiled or dispatched by the production executable.

`checks/state-final.json` compares parent and clean build: all 95 stream
comparisons passed, including single/split counts 1/2/4/5/16/17/32/256 at
P3/P1023/P2049, mixed continuation chains, mid-state hello, reset after a longer
prompt, and continuation after ten rejected requests. IDs, counts and token
ranges were checked. EOF during long decode exited cleanly in 80.91/72.06 ms
(parent/candidate), with no partial reply. Both exits were 0; check elapsed
28.97 seconds. This is protocol-visible state evidence; direct primitive
state bytes are covered by the evolving GDN audit.

Production primitive audits are reproduced after `cargo build --release
--examples` from `engine/`, running each executable under the shared GPU lock:
`linear_input_norm_audit_005 check MODEL_DIR`,
`gdn_input_audit_005 check MODEL_DIR`, and `prepare_fusion_005 check`.
These are development probes; the runtime binary embeds its own kernels and
requires only the model directory.

## Complete candidate versus session 004 parent

`checks/ab-final.json` compared the verified clean-checkout binary with the
unchanged original parent: six alternating pairs at each scored length, one
warmup per length, 256 continuation outputs, normal JSON-lines protocol and
the shared GPU lock. No builds or other GPU probes overlapped this run. All
18 paired token streams were identical; both engines exited 0. The machine
reported no thermal/performance warnings. Elapsed time: 454.41 seconds.

Times below are separate-arm medians; speedups are medians of paired ratios.

| Context | Parent / final decode ms | Paired speedup | Ratio range | Sample SD, percentage points | Wins |
|---:|---:|---:|---:|---:|---:|
| 32768 | 564.247 / 555.857 | 1.030187 | 0.999085–1.031504 | 1.273 | 5/6 |
| 65536 | 749.164 / 728.088 | 1.024557 | 1.015085–1.047087 | 1.080 | 6/6 |
| 131072 | 1102.011 / 1091.802 | 1.014859 | 1.003051–1.025697 | 0.777 | 6/6 |

Task-weighted geometric mean (weights 1/2/3): **1.020628263x**, or **2.06%
higher decode throughput**. This measures the retained mechanisms together
and exceeds the current-session A/A weighted ratio 1.002643. Seventeen of
18 decode pairs improved, including all pairs at the two heaviest lengths.
This is development A/B evidence; formal acceptance remains the harness's.

Prefill parent/final medians: 2720.316/2718.622 ms, 7031.508/7030.860 ms,
20616.853/20631.170 ms at 32K/64K/128K. Paired ratios were
1.002933 / 1.003363 / 1.000531. No prefill algorithm gain is claimed.


## Required public reference evaluation and handoff

Command: `enginesmith dev-eval --no-build --lengths all --rounds 2 --cooldown 15`.
`brief/dev-eval/001.json` reports **PASSED**, complete, no failure, in
235.26 seconds. All 18 controls passed with zero judged mismatches. Sixteen
control streams were identical; control-p2047/p2048 differed as streams but
had zero forced-token mismatches. Timed cases had two rank-2 mismatches among
6,168 total judged steps, both the first output at 32K (one per round, margin
0.0625). No mismatches occurred at 64K or 128K. Each timed case generated 257
outputs, one prefill output plus 256 continuation outputs.

Times below are separate-arm medians; speedups are the reported paired ratios.

| Context | Prefill reference / final ms | Decode reference / final ms | Decode speedup | Mismatches by round |
|---:|---:|---:|---:|---:|
| 32768 | 1662.142 / 2535.115 | 755.052 / 524.276 | 1.440181 | [1, 1] |
| 65536 | 4630.217 / 6531.288 | 972.470 / 704.584 | 1.380204 | [0, 0] |
| 131072 | 14585.563 / 19994.869 | 1411.299 / 1060.882 | 1.330308 | [0, 0] |

Task-weighted decode score: **1.364667242x the reference**; weighted
prefill ratio: **0.709916123x**. Prefill has zero objective
weight and remains slower than the reference. This two-round public result
does not establish the harness's later five-round acceptance, held-out result
or quality scores. The controlled parent A/B establishes this session's
2.06% measured decode improvement; scores from different sessions are not
used as a paired comparison.

Production source changes ended before final timing. The clean binary, full
parent comparison and dev-eval all use identical executable bytes. Only these
notes changed afterward. All engines, probes and benchmarks started by this
session exited; the shared GPU lock was checked available and released.
