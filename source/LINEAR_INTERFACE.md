# Linear kernels

All activations and ordinary gains use native MSL `bfloat`. Packed matrices are
row-major U32 with 8 nibbles/word, scale and bias BF16 per contiguous 64 columns.
All input/output activations are contiguous row-major. Use fastMathEnabled=false.

`linear.metal` requires MSL 3.1. `linear_nax.metal` requires MSL 4.0 and Apple GPU
family 10. Both compile successfully with the installed offline Metal compiler.
Only the NAX route is intended for production prefill. The plain kernel is a
separate numerical comparison route, not an automatically selected fallback.

`LinearShape` is three consecutive uint32 values `{ M, N, K }` (12 bytes):
token rows, output columns, input columns. `NormShape` is `{uint rows, uint cols,
float eps}` (12 bytes). `count` constants are single uint32 values.

| kernel | buffer bindings | threadgroups | threads/group |
|---|---|---|---|
| `embedding_q4` | 0 codes, 1 scales, 2 biases, 3 uint32 tokens[M], 4 output[M,K], 5 LinearShape | ceil(M*K/256),1,1 | 256 |
| `qmv_q4` | 0 codes[N,K/8], 1 scales[N,K/64], 2 biases[N,K/64], 3 x[M,K], 4 y[M,N], 5 LinearShape | M,ceil(N/8),1 | 64 |
| `qmm_q4_nax` | same as qmv | ceil(N/64),ceil(M/64),1 | **128** |
| `qmm_q4_plain` | same as qmv | M,N,1 | 32 |
| `rms_norm_bf16` | 0 x[rows,cols], 1 gain[cols], 2 out[rows,cols], 3 NormShape | rows,1,1 | ceil(cols/128)*32 |
| `add_bf16` | 0 a[count], 1 b[count], 2 out[count], 3 count | ceil(count/256),1,1 | 256 |
| `swiglu_bf16` | 0 gate[count], 1 up[count], 2 out[count], 3 count | ceil(count/256),1,1 | 256 |
| `greedy_bf16` | 0 logits[count], 1 uint32 token[1], 2 count | 1,1,1 | **1024** |

NAX tile: BM=BN=BK=64, four SIMD groups, per-SIMD 32x32 output tile, SK=32,
weight stride padded to 72 BF16. N and M tails are guarded. K must be a multiple
of 64. QMV requires K multiple of 512 (all checkpoint projections satisfy this)
and has four results per SIMD with MLX's 16 values/lane affine-dot order.

RMS supports widths up to 4096 and uses four adjacent values per thread, FP32
statistics and precise reciprocal square root; the normalized input is rounded
to BF16 before gain multiplication. Normal gain is shared across all rows.
All current checkpoint norms use width 128, 256 or 1024. `add_bf16` may alias
either input. `swiglu_bf16` may alias either input. RMS may alias its input.

`greedy_bf16` computes logsumexp with MLX's looped online reduction in one 1024
thread group, rounds the normalizer to BF16, and performs the BF16 subtraction
before choosing the lowest token ID among equal maxima. It therefore accounts
for log-softmax ties. Output token buffer may be used by the next embedding
dispatch without CPU synchronization.

Measured primitive verification used the real Rust/Metal dispatches against
installed MLX 0.32.4, with GPU exclusivity. Five shapes were checked:
`(M,N,K)=(1,64,1024),(5,16,1024),(63,512,1024),(128,1024,3584),
(2048,3584,1024)`. QMV matched all 144 outputs in the first two cases.
NAX matched all 7,340,032 outputs of the 2048-row case. It differed on 5/32,256
and 49/131,072 BF16 values in the 63- and 128-row cases; MLX uses split-K for
these shapes (FP32 partials summed afterward), unlike this initial NAX route.
RMS, dense SwiGLU and embedding each matched all 2,626,560 tested BF16 values.
Greedy matched five random vocabulary rows of length 248,320. These checks do
not replace integrated correctness validation. Performance was not measured.

Reference dispatch source `v0.32.4/mlx/backend/metal/quantized.cpp` chooses QMV
below a device/shape dependent row limit, then split-K matrix multiplication
when multiple K partitions are useful, otherwise NAX. For architecture gen>=17
and size!='d', limits are 33 if K,N<=2048, 25 if K,N<=4096, otherwise 13.
For gen>=15 and size!='d', limits are 13,15,13 respectively. The plain NAX
route is numerically closer to split-K than using QMV in matrix-shaped cases.
For small M>=2, reference may select qmv_wide; the current qmv_q4 shares the
FP32 affine math but does not reproduce its precise reduction tree.

## Reference split-K route

`linear_splitk.metal` uses the vendored MLX BlockMMA simdgroup matrices, a
32x32x32 tile, four SIMD groups, BF16 dequantization and FP32 partial outputs.
Compile with MSL 4.0 for warning-free C++17 support, as for the other libraries.
The host first applies the QMV row limits above. For remaining matrix-shaped
work, calculate `tiles=ceil(M/32)*ceil(N/32)`, then
`splits=min(max(1,512/tiles),K/64)` using integer division. Decrease splits until
`K % (splits*64)==0`. Use NAX if splits==1. Otherwise:

1. Dispatch `qmm_q4_splitk` with buffers 0 codes, 1 scales, 2 biases, 3 x,
   4 FP32 partials; buffer 5 is four uint32 values `{M,N,K,splits}`. Groups are
   `{ceil(N/32),ceil(M/32),splits}` and threads are `{128,1,1}`.
2. Dispatch `qmm_q4_splitk_reduce` with buffers 0 FP32 partials, 1 BF16 y;
   buffer 2 is the same four uint32 parameters. Groups are
   `{ceil(M*N/256),1,1}`, threads `{256,1,1}`.

Scratch is FP32 `[splits,M,N]` (4*splits*M*N bytes). A single reusable 2 MiB
buffer suffices for every shape selected by this split rule, because
`splits*ceil(M/32)*ceil(N/32)<=512`. The reduction adds partials in increasing
split order in FP32 and rounds once to BF16.

Measured GPU comparison to reference split-K yielded **0/366,096** differing
BF16 outputs across `(M,N,K)=(33,16,1024),(25,3584,1024),(13,6144,1024),
(63,512,1024),(128,1024,3584),(2048,16,1024)`. For `(2048,3584,1024)`, NAX
still matches all 7,340,032 outputs while forcing the plain MMA split kernel
with splits=1 differs on 712 values; retain the documented NAX selection.
These were isolated numerical checks, not performance measurements.

`tools/linear_probe.py` is a development utility needing MLX and NumPy.
To build its paired Rust executable, temporarily copy `tools/linear_probe.rs`
to `src/bin/linear_probe.rs`, run `cargo build --release --bin linear_probe`,
then remove the temporary source. The probe holds `/tmp/engine-gpu.lock` and
saves detailed results outside the committed engine directory.
