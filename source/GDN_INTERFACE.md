# GatedDeltaNet Metal kernels

`src/kernels/gdn.metal` is specialized to batch 1, 16 heads, key/value
dimension 128, and convolution width 6144 with four taps. Compile with Metal
3.1 or newer. Each params argument is a four-byte `uint32_t rows`; rows must
be positive. All matrices are row-major, with tokens as the outer dimension.

Call in the order below using a serial compute encoder. Reset both persistent
buffers to zero for a new request: convolution history BF16 `[3,6144]` and
recurrence state FP32 `[16,128_value,128_key]`.

| Kernel | Buffers in index order | Dispatch |
| --- | --- | --- |
| `gdn_conv` | 0 raw projected qkv BF16 `[T,6144]`; 1 history BF16 `[3,6144]`; 2 conv weight BF16 `[6144,4,1]`; 3 output conv BF16 `[T,6144]`; 4 params | grid `[T*6144,1,1]`, threads `[256,1,1]` |
| `gdn_history` | 0 raw projected qkv; 1 history (in-place); 2 params | grid `[6144,1,1]`, threads `[256,1,1]` |
| `gdn_prepare` | 0 conv BF16 `[T,6144]` (Q/K normalized in place); 1 a BF16 `[T,16]`; 2 b BF16 `[T,16]`; 3 A_log FP32 `[16]`; 4 dt_bias BF16 `[16]`; 5 output g FP32 `[T,16]`; 6 output beta BF16 `[T,16]`; 7 params | threadgroups `[T*16,1,1]`, threads `[32,1,1]` |
| `gdn_recur` | 0 normalized conv BF16 `[T,6144]`; 1 g FP32 `[T,16]`; 2 beta BF16 `[T,16]`; 3 state FP32 `[16,128,128]` (in-place); 4 output y BF16 `[T,2048]`; 5 params | grid `[32,16,16]`, threads `[32,2,1]` (equivalently groups `[1,8,16]`) |
| `gdn_norm_gate` | 0 y BF16 `[T,2048]`; 1 z BF16 `[T,2048]`; 2 norm weight BF16 `[128]`; 3 output BF16 `[T,2048]`; 4 params | threadgroups `[T*16,1,1]`, threads `[32,1,1]` |

The raw qkv and convolution buffers must be distinct. History must not be
updated until convolution finishes. The recurrent state is safe to update in
place because each thread owns its 32 elements throughout the sequence.
`gdn_norm_gate` input y/output may alias: each SIMD group owns a whole head,
loads it before stores, and shares no data with other groups.

The normalized conv buffer retains three contiguous 2048-wide Q/K/V blocks
per token; the recurrence reads each with stride 6144. It applies no additional
readout scale. The packed recurrence uses the same eight four-term partials
and explicit shuffle tree as the installed MLX reference.

Numerical bring-up observations: standalone GPU probes against installed
MLX 0.32.4 / mlx-lm 0.32.2 compared 10,000 BF16 points in [-10,10]. The
BF16 sigmoid helper matched all points. Gating matched `compute_g` bitwise
for 10,000 combinations with BF16 dt_bias 0.3 and FP32 A_log in [-2,2].
Additional isolated GPU probes compiled all five complete kernel definitions
successfully and compared their arithmetic against the installed reference:

| Probe | Measured comparison |
| --- | --- |
| Four-tap convolution, 17 rows x 6144 channels, random BF16 input/weight | 0 / 104448 BF16 outputs differ from `mx.conv1d` |
| Q/K normalization, 17 rows x 16 heads x 128 channels | 0 / 34816 BF16 outputs differ for each of Q and K |
| Packed recurrence, 19 rows, random initial FP32 state, normalized random Q/K | 0 / 38912 BF16 outputs and 0 / 262144 FP32 final state elements differ from `gated_delta_kernel` |
| Gated RMS norm, 19 rows x 2048 channels | 0 / 38912 BF16 outputs differ from `precise_swiglu(..., rms_norm(...))` |
| Compiled dense swiglu rounding, 10000 gate points with random BF16 up values | 0 / 10000 BF16 outputs differ |

The initial custom-compiler syntax check concealed a missing native `log1p`
symbol, because MLX supplies that helper in its generated header. Native
startup exposed this; the kernel now contains the MLX `utils.h` calculation
specialized to the softplus exponential's [0,1] range.

The committed reproducer runs the actual native kernels with the host's
`Gpu::compile`, Metal 4.0 and `fastMathEnabled=false`:

```sh
python tools/probe_gdn.py --rows 1,2,19,2048
```

Use an existing development Python with MLX, mlx-lm and NumPy installed.
The script builds `examples/gdn_probe.rs`, takes the shared GPU lock, creates
deterministic fixtures under ignored `target/gdn-probe`, and compares every
native stage against MLX. All eight stages (convolution, history, normalized
QKV, g, beta, state, recurrent output, gated output) had **zero bit differences**
for all four tested row counts. In the 2048-row case that includes 12,582,912
BF16 values each for convolution/QKV, 32,768 each for gates, 262,144 FP32 state
values, and 4,194,304 BF16 values each for recurrent/gated output. Row counts
1 and 2 also verify the short-history shifts using nonzero initial history.

The root host's GDN buffer allocation, parameter order, dispatch dimensions,
state reset, and scratch aliasing were reviewed against this interface with
no discrepancy found. These checks establish isolated native arithmetic
parity; they are not performance measurements or full-model correctness
results.

## Third-party provenance

The packed recurrence is adapted from `mlx_lm/models/gated_delta.py`,
mlx-lm 0.32.2, `_make_gated_delta_packed_kernel`. BF16 elementary-operation
boundaries follow MLX 0.32.4 `unary_ops.h`, `binary_ops.h`, `bf16_math.h`;
the logarithm helper follows `utils.h`.

MIT License

Copyright (c) 2025 Apple Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
