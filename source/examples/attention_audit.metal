/*
Adapted from MLX 0.32.4. Original sources and modifications are recorded in
ATTENTION_INTERFACE.md.

MIT License

Copyright © 2023 Apple Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

*/
// Isolated diagnostic variant. Arithmetic and the 64-way interleaved key
// partition match attention_decode.metal. This source is concatenated after
// that parent source by attention_audit.rs; it uses the parent's definitions.
template <typename T, int D, int HPT, int TK>
[[kernel]] void attention_shared_heads(
    const device T* queries [[buffer(0)]],
    const device T* keys [[buffer(1)]],
    const device T* values [[buffer(2)]],
    device T* out [[buffer(3)]],
    device float* sums [[buffer(4)]],
    device float* maxs [[buffer(5)]],
    constant AttentionDecodeParams& p [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tidtg [[thread_position_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int elements = D / 32;
  typedef float U;
  const int N = int(p.length);
  const int block_idx = tid.z;
  const int base_head = int(tid.x) * 4 + int(tidtg.y) * HPT;
  const size_t head_stride = size_t(p.capacity) * D;
  const int stride = blocks * D;
  keys += size_t(tid.x) * head_stride + block_idx * D + simd_lid * elements;
  values += size_t(tid.x) * head_stride + block_idx * D + simd_lid * elements;
  U q[HPT][elements];
  U o[HPT][elements];
  U max_score[HPT];
  U sum_exp_score[HPT];
  for (int h = 0; h < HPT; h++) {
    const device T* qp = queries + (base_head + h) * D + simd_lid * elements;
    for (int j = 0; j < elements; j++) {
      q[h][j] = static_cast<U>(0.0625f) * qp[j];
      o[h][j] = 0;
    }
    max_score[h] = Limits<U>::finite_min;
    sum_exp_score[h] = 0;
  }
  int i = block_idx;
  for (; i + (TK - 1) * blocks < N; i += blocks * TK) {
    T k_reg[TK][elements];
    T v_reg[TK][elements];
    for (int t = 0; t < TK; t++) {
      const device T* kp = keys + t * stride;
      for (int j = 0; j < elements; j++) k_reg[t][j] = kp[j];
    }
    for (int t = 0; t < TK; t++) {
      const device T* vp = values + t * stride;
      for (int j = 0; j < elements; j++) v_reg[t][j] = vp[j];
    }
    for (int t = 0; t < TK; t++) {
      for (int h = 0; h < HPT; h++) {
        U score = 0;
        for (int j = 0; j < elements; j++) {
          score += q[h][j] * static_cast<U>(k_reg[t][j]);
        }
        score = simd_sum(score);
        U new_max = max(max_score[h], score);
        U factor = fast::exp(max_score[h] - new_max);
        U exp_score = fast::exp(score - new_max);
        max_score[h] = new_max;
        sum_exp_score[h] = sum_exp_score[h] * factor + exp_score;
        for (int j = 0; j < elements; j++) {
          o[h][j] = o[h][j] * factor + exp_score * static_cast<U>(v_reg[t][j]);
        }
      }
    }
    keys += TK * stride;
    values += TK * stride;
  }
  for (; i < N; i += blocks) {
    T k_reg[elements];
    T v_reg[elements];
    for (int j = 0; j < elements; j++) k_reg[j] = keys[j];
    for (int j = 0; j < elements; j++) v_reg[j] = values[j];
    for (int h = 0; h < HPT; h++) {
      U score = 0;
      for (int j = 0; j < elements; j++) score += q[h][j] * k_reg[j];
      score = simd_sum(score);
      U new_max = max(max_score[h], score);
      U factor = fast::exp(max_score[h] - new_max);
      U exp_score = fast::exp(score - new_max);
      max_score[h] = new_max;
      sum_exp_score[h] = sum_exp_score[h] * factor + exp_score;
      for (int j = 0; j < elements; j++) {
        o[h][j] = o[h][j] * factor + exp_score * v_reg[j];
      }
    }
    keys += stride;
    values += stride;
  }
  for (int h = 0; h < HPT; h++) {
    const int row = (base_head + h) * blocks + block_idx;
    if (simd_lid == 0) {
      sums[row] = sum_exp_score[h];
      maxs[row] = max_score[h];
    }
    for (int j = 0; j < elements; j++) {
      out[row * D + simd_lid * elements + j] = static_cast<T>(o[h][j]);
    }
  }
}

#define SHARED(H,T) \
template [[host_name("shared_h" #H "_tk" #T)]] [[kernel]] \
decltype(attention_shared_heads<bfloat,256,H,T>) attention_shared_heads<bfloat,256,H,T>;
SHARED(2,1)
SHARED(2,2)
SHARED(2,4)
SHARED(2,8)
SHARED(4,4)

#define PARENT(T) \
template [[host_name("parent_tk" #T)]] [[kernel]] \
decltype(sdpa_vector_2pass_1<bfloat,256,256,T>) sdpa_vector_2pass_1<bfloat,256,256,T>;
PARENT(1)
PARENT(2)
PARENT(8)

// A lane retains the original block index; a SIMD group owns an eight-element
// output stripe. This directly performs the parent's post-transpose reduction.
// The (block, block+32) additions and subsequent SIMD reduction are unchanged.
[[kernel]] void split2_lanes(
    const device bfloat* partials [[buffer(0)]],
    const device float* sums [[buffer(1)]],
    const device float* maxs [[buffer(2)]],
    device bfloat* out [[buffer(3)]],
    constant AttentionDecodeParams& p [[buffer(4)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  (void)p;
  constexpr int BN = 32;
  constexpr int D = 256;
  constexpr int elem_per_thread = 8;
  typedef float U;
  const int head_idx = tid.x;
  const int stripe = tid.y;
  sums += head_idx * blocks;
  maxs += head_idx * blocks;
  partials += head_idx * blocks * D + lane * D + stripe * elem_per_thread;
  out += head_idx * D + stripe * elem_per_thread;
  U o[elem_per_thread] = {0};
  U sum_exp_score = 0.0;
  U max_score = Limits<U>::finite_min;
  for (int b = 0; b < blocks / BN; ++b) {
    max_score = max(max_score, maxs[lane + BN * b]);
  }
  max_score = simd_max(max_score);
  for (int b = 0; b < blocks / BN; ++b) {
    U factor = fast::exp(maxs[lane + BN * b] - max_score);
    sum_exp_score += factor * sums[lane + BN * b];
  }
  sum_exp_score = simd_sum(sum_exp_score);
  for (int b = 0; b < blocks / BN; ++b) {
    U factor = fast::exp(maxs[lane] - max_score);
    for (int i = 0; i < elem_per_thread; i++) {
      o[i] += factor * static_cast<U>(partials[i]);
    }
    maxs += BN;
    sums += BN;
    partials += BN * D;
  }
  for (int i = 0; i < elem_per_thread; i++) {
    o[i] = simd_sum(o[i]);
    o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
  }
  if (lane == 0) {
    for (int i = 0; i < elem_per_thread; i++) out[i] = static_cast<bfloat>(o[i]);
  }
}

// Isolated fused gate experiment; standalone remap above remains unchanged.
[[kernel]] void split2_lanes_gate(
    const device bfloat* partials [[buffer(0)]],
    const device float* sums [[buffer(1)]],
    const device float* maxs [[buffer(2)]],
    device bfloat* out [[buffer(3)]],
    const device bfloat* projected_q_gate [[buffer(4)]],
    constant AttentionDecodeParams& p [[buffer(5)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  (void)p;
  constexpr int BN = 32;
  constexpr int D = 256;
  constexpr int elem_per_thread = 8;
  typedef float U;
  const int head_idx = tid.x;
  const int stripe = tid.y;
  sums += head_idx * blocks;
  maxs += head_idx * blocks;
  partials += head_idx * blocks * D + lane * D + stripe * elem_per_thread;
  out += head_idx * D + stripe * elem_per_thread;
  U o[elem_per_thread] = {0};
  U sum_exp_score = 0.0;
  U max_score = Limits<U>::finite_min;
  for (int b = 0; b < blocks / BN; ++b) {
    max_score = max(max_score, maxs[lane + BN * b]);
  }
  max_score = simd_max(max_score);
  for (int b = 0; b < blocks / BN; ++b) {
    U factor = fast::exp(maxs[lane + BN * b] - max_score);
    sum_exp_score += factor * sums[lane + BN * b];
  }
  sum_exp_score = simd_sum(sum_exp_score);
  for (int b = 0; b < blocks / BN; ++b) {
    U factor = fast::exp(maxs[lane] - max_score);
    for (int i = 0; i < elem_per_thread; i++) {
      o[i] += factor * static_cast<U>(partials[i]);
    }
    maxs += BN;
    sums += BN;
    partials += BN * D;
  }
  for (int i = 0; i < elem_per_thread; i++) {
    o[i] = simd_sum(o[i]);
    o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
  }
  // All lanes hold every final reduction. Select one coordinate per lane so
  // eight lanes evaluate sigmoid in parallel, preserving the stored BF16
  // attention boundary before multiplication.
  U selected = o[0];
  for (int i = 1; i < elem_per_thread; i++) selected = lane == uint(i) ? o[i] : selected;
  if (lane < elem_per_thread) {
    const bfloat attention = bfloat(selected);
    const uint d = stripe * elem_per_thread + lane;
    float x = float(projected_q_gate[head_idx * 512 + 256 + d]);
    // Exact copy of attn_gate's BF16 sigmoid rounding boundaries.
    float e = float(bfloat(metal::precise::exp(metal::abs(x))));
    float denom = float(bfloat(1.0f + e));
    float inverse = float(bfloat(1.0f / denom));
    bfloat gate = bfloat(x < 0.0f ? inverse : 1.0f - inverse);
    out[lane] = bfloat(float(attention) * float(gate));
  }
}
