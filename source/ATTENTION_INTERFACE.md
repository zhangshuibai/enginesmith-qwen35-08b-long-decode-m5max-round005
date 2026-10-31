# Attention kernel interface

Compile `attention.metal` and `attention_decode.metal` at MSL 3.1; compile
`attention_prefill.metal` at MSL 4.0. All three are independent libraries with
no function constants. Current integration uses fast math disabled.

All strides below are in **elements**. T is query count, offset is the first
query's absolute position, N=offset+T, and C is allocated cache capacity.
Q projection is BF16 `[T,8,512]`, with query then gate inside every head.
Q workspace/output O/gated output are BF16 row-major `[T,8,256]`.
K and V projections are BF16 `[T,2,256]`; each cache is BF16 `[2,C,256]`.

## Preparation and output gate

`AttentionPrepareParams` is three consecutive u32s `{tokens,offset,capacity}`,
12 bytes, with no padding.

| Pipeline | Buffer indices | Threadgroups | Threads per group |
|---|---|---|---|
| `attn_prepare_q` | 0 q_projection, 1 q_norm_gain[256], 2 Q, 3 params | `(8,T,1)` | `(64,1,1)` |
| `attn_prepare_kv` | 0 k_projection, 1 v_projection, 2 k_norm_gain[256], 3 Kcache, 4 Vcache, 5 params | `(2,T,1)` | `(64,1,1)` |
| `attn_gate` | 0 O, 1 q_projection, 2 gated, 3 u32 T | ceil(T*2048/256) | 256 |

Gate may write in place over O. Preparation writes Q and all newly projected
K/V before attention starts. Q/K RMS uses 4 contiguous elements per thread,
64 threads, FP32 accumulation, BF16 normalization then BF16 gain product.
Rotary pairs are (i,i+32), i<32, with reference exp2 frequency and fast sin/cos.
Sigmoid has BF16 rounding after exp(abs(x)), addition, reciprocal and positive
branch subtraction, followed by a separately rounded output product.

## NAX prefill

`attn_prefill` buffers: **0 Q, 1 Kcache, 2 Vcache, 3 O, 4 AttnParams**.
The generated template also declares unused optional mask/sink/lse buffers;
these have been specialized away and need no bindings.

Dispatch threadgroups `(ceil(T/64),8,1)` with `(256,1,1)` threads.
Shape is BQ=64, BK=32, BD=256, WM=4, WN=2, matching MLX 0.32.4's
head-dimension-split route for this checkpoint's normal 2048-token chunks.

`AttnParams` is `#[repr(C)]`, **152 bytes**, alignment 8:

```rust
struct AttnParams {
    b: i32, h: i32, d: i32,
    q_l: i32, k_l: i32,
    gqa_factor: i32, scale: f32,
    n_q: i32, n_k: i32,
    n_q_aligned: i32, n_k_aligned: i32,
    q_l_rem: i32, k_l_rem: i32, q_l_off: i32,
    q_strides: [i64;3],
    k_strides: [i64;3],
    v_strides: [i64;3],
    o_strides: [i64;3],
}
```

Set fields in order:

```text
b=1, h=8, d=256, q_l=T, k_l=N,
gqa_factor=4, scale=0.0625,
n_q=(T+63)/64, n_k=(N+31)/32,
n_q_aligned=T/64, n_k_aligned=N/32,
q_l_rem=T%64, k_l_rem=N%32, q_l_off=offset,
q_strides=[T*2048, 256, 2048],
k_strides=[2*C*256, C*256, 256],
v_strides=[2*C*256, C*256, 256],
o_strides=[T*2048, 256, 2048].
```

The compiled route always enables causal masking and safe final-tile loads.
No dynamic threadgroup-memory argument is needed.

## Decode

`AttentionDecodeParams` is two consecutive u32s `{length,capacity}`, 8 bytes.
Length is the total live KV length N, including the current query token.
All paths here require **T=1**. Fixed GQA=4, D=256, scale=1/16.

| Pipeline | Buffer indices | Threadgroups | Threads per group |
|---|---|---|---|
| `attn_decode` | 0 Q, 1 Kcache, 2 Vcache, 3 O, 4 params | `(8,1,1)` | `(1024,1,1)` |
| `attn_decode_split1` | 0 Q, 1 Kcache, 2 Vcache, 3 partials, 4 sums, 5 maxs, 6 params | `(2,1,64)` | **`(32,4,1)`** |
| `attn_decode_split2` (parent/probe) | 0 partials, 1 sums, 2 maxs, 3 O, 4 params | `(8,1,1)` | `(1024,1,1)` |
| `attn_decode_split2_gate` | 0 partials, 1 sums, 2 maxs, 3 gated, 4 q_projection, 5 params | `(8,32,1)` | `(32,1,1)` |

Session 002 production uses the fused reduction/gate for N>=1024 and skips
the separate `attn_gate` only on that route. A SIMD group owns eight output
dimensions; lane `b` adds blocks `b` and `b+32` in the parent order before the
same SIMD sum. This removes the threadgroup transpose/barriers. The attention
result is rounded to BF16 before the unchanged BF16 sigmoid/product graph.
See NOTES.md for byte comparisons and paired performance measurements.

Split stage 1 must use the stated three-dimensional thread shape: tidtg.y is
query head within the KV group. It is the ordinary MLX two-pass kernel (not
its optional duplication-free GQA kernel), so grid x is **two KV heads**,
and four simdgroups compute the four query heads.

Scratch: partials BF16 `[8,64,256]` (262144 bytes), sums FP32 `[8,64]`
(2048 bytes), maxs FP32 `[8,64]` (2048 bytes). Scratch can be reused after the
combine stage. MLX's M5 architecture suffix `s` uses 64 blocks for GQA4/T1,
with the two-pass route starting at N>=1024; the native numerical probe verified the actual device architecture as
`applegpu_g17s`. For suffix `d`, reference uses 128 blocks,
which is not currently instantiated.

## Provenance and validation

`attention_generate.py` is a development-only flattening/specialization tool,
not part of the build or runtime. It takes the MLX include directory and MIT
license as arguments and emits standalone `.metal` sources. The source tree
retains the full MIT notice. Original installed headers are MLX 0.32.4:

- `mlx/backend/metal/kernels/sdpa_vector.h`
- `mlx/backend/metal/kernels/steel/attn/kernels/steel_attention_nax.h`
- Its recursive steel header dependencies, annotated in the generated source.

The original decode mathematical bodies remain for probes; the session 002
fused reduction/gate is described above. Original signatures are simplified to
model-specific structs and masks/sinks are statically disabled. Prefill uses
static function constants and one explicit template instantiation. Q/K norm,
rotary arithmetic and sigmoid follow MLX 0.32.4's rms_norm.metal, rope.metal,
and unary_ops.h. Custom fusion preserves the documented BF16 boundaries.

All three Metal sources compile cleanly with `-fno-fast-math`. Native Metal
versus installed MLX numerical probes gave **zero bit differences** for Q/K/V
and full attention output/gate at decode lengths 1,31,1023,1024,4096,131072 and
NAX prefill (T,N)=(1024,1024),(2047,4095). This includes over six million
prefill output values. At (33,33), using NAX instead of the reference's unfused
route differed on 32141/67584 attention elements, maximum absolute error
0.015625. These are measured kernel comparisons, not end-to-end gate results.
Performance has not been separately measured. The reference
uses an unfused BF16 attention route for 9<=T<1024 at D256; NAX can be used
there but is a distinct numerical route requiring validation.


## Short prefill reference routes

`attn_small` lives in `attention_decode.metal`, uses the same four buffers and
8-byte `AttentionDecodeParams` as `attn_decode`, and dispatches `(8,T,1)` groups
of `(1024,1,1)` threads. Use for **2<=T<=8 and N<1024**. It is the reference
single-pass vector implementation with causal masking and token-major output.
For larger N this implementation is mathematically valid but the reference
switches to a different two-pass route and block count.

For **9<=T<1024 and N<=4096**, three kernels in `attention.metal` explicitly
materialize BF16 scores and BF16 probabilities, following the unfused reference
graph. Allocate scores scratch **T*8*N*2 bytes**, laid out `[T,8,N]`.
`AttentionShortParams` is four consecutive u32s `{tokens,length,capacity,offset}`
(16 bytes). All three accept this same struct after their buffers.

| Pipeline | Buffer indices | Threadgroups | Threads per group |
|---|---|---|---|
| `attn_short_scores` | 0 Q, 1 Kcache, 2 scores, 3 params | `(ceil(N/4),8,T)` | `(128,1,1)` |
| `attn_short_softmax` | 0 scores (in-place), 1 params | `(T*8,1,1)` | `(ceil(N/128)*32,1,1)` |
| `attn_short_values` | 0 probabilities (same scores), 1 Vcache, 2 O, 3 params | `(8,T,1)` | `(256,1,1)` |

These kernels preserve all BF16 graph boundaries. QK and PV are scalar/SIMD
implementations rather than the reference's general matrix kernels, so their
reduction order is independently validated. Softmax preserves the reference's
four-values-per-thread reduction. N>4096 is unsupported by this softmax launch.
Measured native-versus-MLX output comparisons for the BF16 fallback:

| (T,N) | Different output elements / total | Maximum absolute difference |
|---|---:|---:|
| (16,16) | 0 / 32768 | 0 |
| (33,33) | 3 / 67584 | 0.00000190735 |
| (127,127) | 5 / 260096 | 0.00048828125 |
| (511,511) | 511 / 1046528 | 0.001953125 |
| (1023,1023) | 1825 / 2095104 | 0.00244140625 |
| (33,3000) | 132 / 67584 | 0.000244140625 |

Q/K/V were bitwise exact in all cases. `attn_small` was bitwise exact for
Q/K/V/output/gate at (T,N)=(2,2),(3,3),(8,8),(8,513). Run the reproducible
native comparison through `tools/attention_probe.py --cases T:N,...`; it builds
only `cargo --example attention_probe`, acquires `/tmp/engine-gpu.lock`, and
keeps generated fixtures/results in the session directory outside `engine/`.
The default suite covers decode, small causal, unfused short and NAX prefill
routes. This Python/MLX development tool is never loaded by the engine.

Upstream arithmetic/dispatch sources inspected:

- https://github.com/ml-explore/mlx/blob/v0.32.4/mlx/backend/metal/kernels/rms_norm.metal
- https://github.com/ml-explore/mlx/blob/v0.32.4/mlx/backend/metal/kernels/rope.metal
- https://github.com/ml-explore/mlx/blob/v0.32.4/mlx/backend/metal/scaled_dot_product_attention.cpp
- https://github.com/ml-explore/mlx/blob/v0.32.4/mlx/fast.cpp

