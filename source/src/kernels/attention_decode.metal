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
#include <metal_stdlib>
using namespace metal;
#ifndef METAL_FUNC
#define METAL_FUNC inline
#endif

template <typename T> struct Limits;
template <> struct Limits<float> { static constexpr constant float finite_min = -3.4028234663852886e38f; };
template <> struct Limits<bfloat> { static constexpr constant float finite_min = -3.3895313892515355e38f; };
struct AttentionDecodeParams { uint length; uint capacity; };
// Copyright © 2024 Apple Inc.

#include <metal_simdgroup>

using namespace metal;

constant bool query_transposed = true;
constant bool do_causal = false;
constant bool bool_mask = false;
constant bool float_mask = false;
constant bool has_sinks = false;
constant int blocks = 64;

template <typename T, int D, int V = D>
[[kernel]] void sdpa_vector(
    const device T* queries [[buffer(0)]],
    const device T* keys [[buffer(1)]],
    const device T* values [[buffer(2)]],
    device T* out [[buffer(3)]],
    constant AttentionDecodeParams& p [[buffer(4)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  const int N = int(p.length);
  const size_t k_head_stride = size_t(p.capacity) * 256;
  const size_t k_seq_stride = 256;
  const size_t v_head_stride = size_t(p.capacity) * 256;
  const size_t v_seq_stride = 256;
  const float scale = 0.0625f;
  const device bool* bmask = nullptr;
  const device T* fmask = nullptr;
  const device T* sinks = nullptr;
  const int mask_kv_seq_stride = 0;
  const int mask_q_seq_stride = 0;
  const int mask_head_stride = 0;

  const int gqa_factor = 4;
  const int num_q_heads = 8;

  constexpr int BN = 32;
  constexpr int BD = 32;
  constexpr int qk_per_thread = D / BD;
  constexpr int v_per_thread = V / BD;
  int inner_k_stride = BN * int(k_seq_stride);
  int inner_v_stride = BN * int(v_seq_stride);

  typedef float U;

  thread U q[qk_per_thread];
  thread U k[qk_per_thread];
  thread U o[v_per_thread];

  threadgroup U outputs[BN * BD];
  threadgroup U max_scores[BN];
  threadgroup U sum_exp_scores[BN];

  // Adjust positions
  const int q_batch_head_idx = tid.x;
  const int q_seq_idx = tid.y;
  const int kv_head_idx = q_batch_head_idx / gqa_factor;
  const int kv_seq_idx = simd_gid;
  const int o_offset = q_batch_head_idx * tpg.y + q_seq_idx;
  const int q_offset =
      query_transposed ? tpg.x * q_seq_idx + q_batch_head_idx : o_offset;
  queries += q_offset * D + simd_lid * qk_per_thread;
  keys += kv_head_idx * k_head_stride + simd_gid * k_seq_stride +
      simd_lid * qk_per_thread;
  values += kv_head_idx * v_head_stride + simd_gid * v_seq_stride +
      simd_lid * v_per_thread;
  if (bool_mask) {
    bmask += q_batch_head_idx * mask_head_stride +
        kv_seq_idx * mask_kv_seq_stride + q_seq_idx * mask_q_seq_stride;
  }
  if (float_mask) {
    fmask += q_batch_head_idx * mask_head_stride +
        kv_seq_idx * mask_kv_seq_stride + q_seq_idx * mask_q_seq_stride;
  }

  out += o_offset * V + simd_gid * v_per_thread;

  // Read the query and 0 the output accumulator
  for (int i = 0; i < qk_per_thread; i++) {
    q[i] = static_cast<U>(scale) * queries[i];
  }
  for (int i = 0; i < v_per_thread; i++) {
    o[i] = 0;
  }

  U max_score = Limits<U>::finite_min;
  U sum_exp_score = 0;
  if (has_sinks && simd_gid == 0) {
    max_score = static_cast<U>(sinks[q_batch_head_idx % num_q_heads]);
    sum_exp_score = 1;
  }

  // For each key
  for (int i = simd_gid; i < N; i += BN) {
    bool use_key = true;
    if (do_causal) {
      use_key = i <= (N - int(tpg.y) + int(q_seq_idx));
    } else if (bool_mask) {
      use_key = bmask[0];
    } else if (float_mask) {
      use_key = (fmask[0] >= Limits<T>::finite_min);
    }
    if (use_key) {
      // Read the key
      for (int j = 0; j < qk_per_thread; j++) {
        k[j] = keys[j];
      }

      // Compute the i-th score
      U score = 0;
      for (int j = 0; j < qk_per_thread; j++) {
        score += q[j] * k[j];
      }
      score = simd_sum(score);
      if (float_mask) {
        score += static_cast<U>(fmask[0]);
      }

      // Update the accumulators
      U new_max = max(max_score, score);
      U factor = fast::exp(max_score - new_max);
      U exp_score = fast::exp(score - new_max);

      max_score = new_max;
      sum_exp_score = sum_exp_score * factor + exp_score;

      // Update the output accumulator
      for (int j = 0; j < v_per_thread; j++) {
        o[j] = o[j] * factor + exp_score * values[j];
      }
    }

    // Move the pointers to the next kv
    keys += inner_k_stride;
    values += inner_v_stride;
    if (bool_mask) {
      bmask += BN * mask_kv_seq_stride;
    }
    if (float_mask) {
      fmask += BN * mask_kv_seq_stride;
    }
  }

  // Each thread has a partial part of the output so we need to combine them.

  // First let's communicate the max and sum_exp
  if (simd_lid == 0) {
    max_scores[simd_gid] = max_score;
    sum_exp_scores[simd_gid] = sum_exp_score;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  max_score = max_scores[simd_lid];
  U new_max = simd_max(max_score);
  U factor = fast::exp(max_score - new_max);
  sum_exp_score = simd_sum(sum_exp_scores[simd_lid] * factor);

  // Now we need to aggregate all the outputs
  for (int i = 0; i < v_per_thread; i++) {
    outputs[simd_lid * BD + simd_gid] = o[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    o[i] = simd_sum(outputs[simd_gid * BD + simd_lid] * factor);
    o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // And write the output
  if (simd_lid == 0) {
    for (int i = 0; i < v_per_thread; i++) {
      out[i] = static_cast<T>(o[i]);
    }
  }
}

template <typename T, int D, int V = D, int TK = 4>
[[kernel]] void sdpa_vector_2pass_1(
    const device T* queries [[buffer(0)]],
    const device T* keys [[buffer(1)]],
    const device T* values [[buffer(2)]],
    device T* out [[buffer(3)]],
    device float* sums [[buffer(4)]],
    device float* maxs [[buffer(5)]],
    constant AttentionDecodeParams& p [[buffer(6)]],
    uint3 tptg [[threads_per_threadgroup]],
    uint3 tidtg [[thread_position_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  const int N = int(p.length);
  const size_t k_head_stride = size_t(p.capacity) * 256;
  const size_t k_seq_stride = 256;
  const size_t v_head_stride = size_t(p.capacity) * 256;
  const size_t v_seq_stride = 256;
  const float scale = 0.0625f;
  const device bool* bmask = nullptr;
  const device T* fmask = nullptr;
  const device T* sinks = nullptr;
  const int mask_kv_seq_stride = 0;
  const int mask_q_seq_stride = 0;
  const int mask_head_stride = 0;

  constexpr int BD = 32;
  constexpr int qk_per_thread = D / BD;
  constexpr int v_per_thread = V / BD;
  static_assert(sizeof(T) == 2 && qk_per_thread == 8 && v_per_thread == 8,
                "vector loads require eight 16-bit elements per lane");

  typedef float U;

  thread U q[qk_per_thread];
  thread U o[v_per_thread] = {0};

  // Adjust positions
  const int kv_head_idx = tid.x;
  const int batch_idx = tid.y;
  const int block_idx = tid.z;
  const int gqa_factor = tptg.y;
  const int q_seq_len = tptg.z;
  const int q_seq_idx = tidtg.z;
  const int q_head_idx = gqa_factor * kv_head_idx + tidtg.y;
  const int num_kv_heads = tpg.x;
  const int num_q_heads = num_kv_heads * gqa_factor;
  const int q_batch_head_idx = (batch_idx * num_q_heads + q_head_idx);
  const int o_offset = q_batch_head_idx * q_seq_len + q_seq_idx;
  const int q_offset =
      query_transposed ? num_q_heads * q_seq_idx + q_batch_head_idx : o_offset;

  queries += q_offset * D + simd_lid * qk_per_thread;

  const int kv_batch_head_idx = batch_idx * num_kv_heads + kv_head_idx;
  keys += kv_batch_head_idx * k_head_stride + block_idx * k_seq_stride +
      simd_lid * qk_per_thread;
  values += kv_batch_head_idx * v_head_stride + block_idx * v_seq_stride +
      simd_lid * v_per_thread;
  out += o_offset * blocks * V + block_idx * V + simd_lid * v_per_thread;
  if (bool_mask) {
    bmask += q_batch_head_idx * mask_head_stride +
        block_idx * mask_kv_seq_stride + q_seq_idx * mask_q_seq_stride;
  }
  if (float_mask) {
    fmask += q_batch_head_idx * mask_head_stride +
        block_idx * mask_kv_seq_stride + q_seq_idx * mask_q_seq_stride;
  }
  sums += o_offset * blocks + block_idx;
  maxs += o_offset * blocks + block_idx;

  // Read the query
  for (int i = 0; i < qk_per_thread; i++) {
    q[i] = static_cast<U>(scale) * queries[i];
  }

  U max_score = Limits<U>::finite_min;
  U sum_exp_score = 0;
  if (has_sinks && block_idx == 0) {
    max_score = static_cast<U>(sinks[q_head_idx]);
    sum_exp_score = 1;
  }

  const int stride_k_tok = blocks * int(k_seq_stride);
  const int stride_v_tok = blocks * int(v_seq_stride);
  const int stride_mask_tok =
      (bool_mask || float_mask) ? blocks * mask_kv_seq_stride : 0;

  // For each key
  int i = block_idx;
  for (; i + (TK - 1) * blocks < N; i += blocks * TK) {
    // Prefetch: issue all TK K/V loads first so the HW can overlap them.
    T k_reg[TK][qk_per_thread];
    T v_reg[TK][v_per_thread];
    // Each lane owns eight contiguous BF16 elements, aligned to 16 bytes.
    // Explicit vector loads retain every value and arithmetic order while
    // expressing each K/V row as one 16-byte load per lane.
    for (int t = 0; t < TK; t++) {
      const device T* kp = keys + t * stride_k_tok;
      uint4 packed = reinterpret_cast<const device uint4*>(kp)[0];
      vec<T, 4> lo = as_type<vec<T, 4>>(packed.xy);
      vec<T, 4> hi = as_type<vec<T, 4>>(packed.zw);
      for (int j = 0; j < 4; j++) {
        k_reg[t][j] = lo[j];
        k_reg[t][j + 4] = hi[j];
      }
    }
    for (int t = 0; t < TK; t++) {
      const device T* vp = values + t * stride_v_tok;
      uint4 packed = reinterpret_cast<const device uint4*>(vp)[0];
      vec<T, 4> lo = as_type<vec<T, 4>>(packed.xy);
      vec<T, 4> hi = as_type<vec<T, 4>>(packed.zw);
      for (int j = 0; j < 4; j++) {
        v_reg[t][j] = lo[j];
        v_reg[t][j + 4] = hi[j];
      }
    }

    for (int t = 0; t < TK; t++) {
      int ii = i + t * blocks;
      bool use_key = true;
      if (do_causal) {
        use_key = ii <= (N - q_seq_len + int(q_seq_idx));
      } else if (bool_mask) {
        use_key = bmask[t * stride_mask_tok];
      } else if (float_mask) {
        use_key = (fmask[t * stride_mask_tok] >= Limits<T>::finite_min);
      }
      if (use_key) {
        U score = 0;
        for (int j = 0; j < qk_per_thread; j++) {
          score += q[j] * static_cast<U>(k_reg[t][j]);
        }
        score = simd_sum(score);
        if (float_mask) {
          score += static_cast<U>(fmask[t * stride_mask_tok]);
        }

        U new_max = max(max_score, score);
        U factor = fast::exp(max_score - new_max);
        U exp_score = fast::exp(score - new_max);
        max_score = new_max;
        sum_exp_score = sum_exp_score * factor + exp_score;
        for (int j = 0; j < v_per_thread; j++) {
          o[j] = o[j] * factor + exp_score * static_cast<U>(v_reg[t][j]);
        }
      }
    }

    keys += TK * stride_k_tok;
    values += TK * stride_v_tok;
    if (bool_mask) {
      bmask += TK * stride_mask_tok;
    }
    if (float_mask) {
      fmask += TK * stride_mask_tok;
    }
  }

  // Tail (fewer than TK remaining keys)
  for (; i < N; i += blocks) {
    bool use_key = true;
    if (do_causal) {
      use_key = i <= (N - q_seq_len + int(q_seq_idx));
    } else if (bool_mask) {
      use_key = bmask[0];
    } else if (float_mask) {
      use_key = (fmask[0] >= Limits<T>::finite_min);
    }
    if (use_key) {
      // Compute the i-th score
      U score = 0;
      for (int j = 0; j < qk_per_thread; j++) {
        score += q[j] * keys[j];
      }
      score = simd_sum(score);

      if (float_mask) {
        score += fmask[0];
      }

      // Update the accumulators
      U new_max = max(max_score, score);
      U factor = fast::exp(max_score - new_max);
      U exp_score = fast::exp(score - new_max);

      max_score = new_max;
      sum_exp_score = sum_exp_score * factor + exp_score;

      // Update the output accumulator
      for (int j = 0; j < v_per_thread; j++) {
        o[j] = o[j] * factor + exp_score * values[j];
      }
    }

    // Move the pointers to the next kv
    keys += stride_k_tok;
    values += stride_v_tok;
    if (bool_mask) {
      bmask += stride_mask_tok;
    }
    if (float_mask) {
      fmask += stride_mask_tok;
    }
  }

  // Write the sum and max and outputs
  if (simd_lid == 0) {
    sums[0] = sum_exp_score;
    maxs[0] = max_score;
  }

  for (int i = 0; i < v_per_thread; i++) {
    out[i] = static_cast<T>(o[i]);
  }
}

// Duplication-free variant for high gqa_factor decode: each simdgroup owns a
// contiguous token sub-chunk and computes HPT of its group's query heads, so
// each K/V byte is read G / HPT times instead of G times. Single-token
// queries without mask or sinks only; the partials layout matches
// sdpa_vector_2pass_2.
template <typename T, int D, int V, int G, int HPT>
[[kernel]] void sdpa_vector_2pass_1_gqa(
    const device T* queries [[buffer(0)]],
    const device T* keys [[buffer(1)]],
    const device T* values [[buffer(2)]],
    device T* out [[buffer(3)]],
    device float* sums [[buffer(4)]],
    device float* maxs [[buffer(5)]],
    const constant int& N [[buffer(7)]],
    const constant size_t& k_head_stride [[buffer(8)]],
    const constant size_t& k_seq_stride [[buffer(9)]],
    const constant size_t& v_head_stride [[buffer(10)]],
    const constant size_t& v_seq_stride [[buffer(11)]],
    const constant float& scale [[buffer(12)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint3 tidtg [[thread_position_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BD = 32;
  constexpr int qk_per_thread = D / BD;
  constexpr int v_per_thread = V / BD;
  constexpr int NT = G / HPT;

  typedef float U;

  const int kv_head_idx = tid.x;
  const int batch_idx = tid.y;
  const int block_idx = tid.z;
  const int blocks = tpg.z;
  const int g = tidtg.y;
  const int cchunk = g / NT;
  const int h0 = (g % NT) * HPT;
  const int num_kv_heads = tpg.x;
  const int num_q_heads = num_kv_heads * G;
  const int base_head = batch_idx * num_q_heads + kv_head_idx * G;
  const int kv_batch_head_idx = batch_idx * num_kv_heads + kv_head_idx;

  const int chunk = (N + blocks - 1) / blocks;
  const int kstart = block_idx * chunk;
  const int kend = min(N, kstart + chunk);
  const int sub = (chunk + HPT - 1) / HPT;
  const int s0 = kstart + cchunk * sub;
  const int s1 = min(kend, s0 + sub);

  const device T* kp = keys + kv_batch_head_idx * k_head_stride +
      s0 * k_seq_stride + simd_lid * qk_per_thread;
  const device T* vp = values + kv_batch_head_idx * v_head_stride +
      s0 * v_seq_stride + simd_lid * v_per_thread;

  U q[HPT][qk_per_thread];
  for (int j = 0; j < HPT; j++) {
    const device T* qp =
        queries + (base_head + h0 + j) * D + simd_lid * qk_per_thread;
    for (int i = 0; i < qk_per_thread; i++) {
      q[j][i] = static_cast<U>(scale) * qp[i];
    }
  }

  U max_score[HPT];
  U sum_exp_score[HPT];
  U o[HPT][v_per_thread];
  for (int j = 0; j < HPT; j++) {
    max_score[j] = Limits<U>::finite_min;
    sum_exp_score[j] = 0;
    for (int i = 0; i < v_per_thread; i++) {
      o[j][i] = 0;
    }
  }

  for (int t = s0; t < s1; t++) {
    U kr[qk_per_thread];
    U vr[v_per_thread];
    for (int i = 0; i < qk_per_thread; i++) {
      kr[i] = kp[i];
    }
    for (int i = 0; i < v_per_thread; i++) {
      vr[i] = vp[i];
    }
    kp += k_seq_stride;
    vp += v_seq_stride;
    for (int j = 0; j < HPT; j++) {
      U score = 0;
      for (int i = 0; i < qk_per_thread; i++) {
        score += q[j][i] * kr[i];
      }
      score = simd_sum(score);
      U new_max = max(max_score[j], score);
      U factor = fast::exp(max_score[j] - new_max);
      U exp_score = fast::exp(score - new_max);
      max_score[j] = new_max;
      sum_exp_score[j] = sum_exp_score[j] * factor + exp_score;
      for (int i = 0; i < v_per_thread; i++) {
        o[j][i] = o[j][i] * factor + exp_score * vr[i];
      }
    }
  }

  threadgroup U o_sh[G * HPT * V];
  threadgroup U se_sh[G * HPT];
  threadgroup U mx_sh[G * HPT];
  for (int j = 0; j < HPT; j++) {
    int slot = (h0 + j) * HPT + cchunk;
    U inv = sum_exp_score[j] > 0 ? 1 / sum_exp_score[j] : 0;
    for (int i = 0; i < v_per_thread; i++) {
      o_sh[slot * V + simd_lid * v_per_thread + i] = o[j][i] * inv;
    }
    if (simd_lid == 0) {
      se_sh[slot] = sum_exp_score[j];
      mx_sh[slot] = max_score[j];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  U gmax = Limits<U>::finite_min;
  for (int s = 0; s < HPT; s++) {
    gmax = max(gmax, mx_sh[g * HPT + s]);
  }
  U denom = 0;
  U acc[v_per_thread] = {0};
  for (int s = 0; s < HPT; s++) {
    U w = se_sh[g * HPT + s] * fast::exp(mx_sh[g * HPT + s] - gmax);
    denom += w;
    for (int i = 0; i < v_per_thread; i++) {
      acc[i] += w * o_sh[(g * HPT + s) * V + simd_lid * v_per_thread + i];
    }
  }

  const int o_offset = base_head + g;
  device T* op =
      out + o_offset * blocks * V + block_idx * V + simd_lid * v_per_thread;
  for (int i = 0; i < v_per_thread; i++) {
    op[i] = static_cast<T>(acc[i]);
  }
  if (simd_lid == 0) {
    sums[o_offset * blocks + block_idx] = denom;
    maxs[o_offset * blocks + block_idx] = gmax;
  }
}

template <typename T, int D>
[[kernel]] void sdpa_vector_2pass_2(
    const device T* partials [[buffer(0)]],
    const device float* sums [[buffer(1)]],
    const device float* maxs [[buffer(2)]],
    device T* out [[buffer(3)]],
    constant AttentionDecodeParams& p [[buffer(4)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  (void)p;

  constexpr int BN = 32;
  constexpr int BD = 32;
  constexpr int elem_per_thread = D / BD;

  typedef float U;

  thread U o[elem_per_thread] = {0};
  threadgroup U outputs[BN * BD];

  // Adjust positions
  const int head_idx = tid.x;
  const int q_seq_idx = tid.y;
  const int q_offset = head_idx * tpg.y + q_seq_idx;
  partials += q_offset * blocks * D + simd_gid * D + simd_lid * elem_per_thread;
  sums += q_offset * blocks;
  maxs += q_offset * blocks;
  out += q_offset * D + simd_gid * elem_per_thread;

  // Set defaults
  U sum_exp_score = 0.0;
  U max_score = Limits<U>::finite_min;

  // Reduce the max
  for (int b = 0; b < blocks / BN; ++b) {
    max_score = max(max_score, maxs[simd_lid + BN * b]);
  }
  max_score = simd_max(max_score);

  // Reduce the d
  for (int b = 0; b < blocks / BN; ++b) {
    U factor = fast::exp(maxs[simd_lid + BN * b] - max_score);
    sum_exp_score += factor * sums[simd_lid + BN * b];
  }
  sum_exp_score = simd_sum(sum_exp_score);

  // Reduce the sum exp and partials
  for (int b = 0; b < blocks / BN; ++b) {
    U factor = fast::exp(maxs[simd_gid] - max_score);

    // Update the output accumulator
    for (int i = 0; i < elem_per_thread; i++) {
      o[i] += factor * static_cast<U>(partials[i]);
    }
    maxs += BN;
    sums += BN;
    partials += BN * D;
  }

  // Use shared memory to transpose and reduce the final block
  for (int i = 0; i < elem_per_thread; i++) {
    outputs[simd_lid * BD + simd_gid] = o[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    o[i] = simd_sum(outputs[simd_gid * BD + simd_lid]);
    o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // And write the output
  if (simd_lid == 0) {
    for (int i = 0; i < elem_per_thread; i++) {
      out[i] = static_cast<T>(o[i]);
    }
  }
}
template <typename T, int D, int V = D>
[[kernel]] void sdpa_vector_small(
    const device T* queries [[buffer(0)]],
    const device T* keys [[buffer(1)]],
    const device T* values [[buffer(2)]],
    device T* out [[buffer(3)]],
    constant AttentionDecodeParams& p [[buffer(4)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  const int N = int(p.length);
  const size_t k_head_stride = size_t(p.capacity) * 256;
  const size_t k_seq_stride = 256;
  const size_t v_head_stride = size_t(p.capacity) * 256;
  const size_t v_seq_stride = 256;
  const float scale = 0.0625f;
  const device bool* bmask = nullptr;
  const device T* fmask = nullptr;
  const device T* sinks = nullptr;
  const int mask_kv_seq_stride = 0;
  const int mask_q_seq_stride = 0;
  const int mask_head_stride = 0;

  const int gqa_factor = 4;
  const int num_q_heads = 8;

  constexpr int BN = 32;
  constexpr int BD = 32;
  constexpr int qk_per_thread = D / BD;
  constexpr int v_per_thread = V / BD;
  int inner_k_stride = BN * int(k_seq_stride);
  int inner_v_stride = BN * int(v_seq_stride);

  typedef float U;

  thread U q[qk_per_thread];
  thread U k[qk_per_thread];
  thread U o[v_per_thread];

  threadgroup U outputs[BN * BD];
  threadgroup U max_scores[BN];
  threadgroup U sum_exp_scores[BN];

  // Adjust positions
  const int q_batch_head_idx = tid.x;
  const int q_seq_idx = tid.y;
  const int kv_head_idx = q_batch_head_idx / gqa_factor;
  const int kv_seq_idx = simd_gid;
  const int o_offset = q_seq_idx * tpg.x + q_batch_head_idx;
  const int q_offset =
      query_transposed ? tpg.x * q_seq_idx + q_batch_head_idx : o_offset;
  queries += q_offset * D + simd_lid * qk_per_thread;
  keys += kv_head_idx * k_head_stride + simd_gid * k_seq_stride +
      simd_lid * qk_per_thread;
  values += kv_head_idx * v_head_stride + simd_gid * v_seq_stride +
      simd_lid * v_per_thread;
  if (bool_mask) {
    bmask += q_batch_head_idx * mask_head_stride +
        kv_seq_idx * mask_kv_seq_stride + q_seq_idx * mask_q_seq_stride;
  }
  if (float_mask) {
    fmask += q_batch_head_idx * mask_head_stride +
        kv_seq_idx * mask_kv_seq_stride + q_seq_idx * mask_q_seq_stride;
  }

  out += o_offset * V + simd_gid * v_per_thread;

  // Read the query and 0 the output accumulator
  for (int i = 0; i < qk_per_thread; i++) {
    q[i] = static_cast<U>(scale) * queries[i];
  }
  for (int i = 0; i < v_per_thread; i++) {
    o[i] = 0;
  }

  U max_score = Limits<U>::finite_min;
  U sum_exp_score = 0;
  if (has_sinks && simd_gid == 0) {
    max_score = static_cast<U>(sinks[q_batch_head_idx % num_q_heads]);
    sum_exp_score = 1;
  }

  // For each key
  for (int i = simd_gid; i < N; i += BN) {
    bool use_key = true;
    if (true) {
      use_key = i <= (N - int(tpg.y) + int(q_seq_idx));
    } else if (bool_mask) {
      use_key = bmask[0];
    } else if (float_mask) {
      use_key = (fmask[0] >= Limits<T>::finite_min);
    }
    if (use_key) {
      // Read the key
      for (int j = 0; j < qk_per_thread; j++) {
        k[j] = keys[j];
      }

      // Compute the i-th score
      U score = 0;
      for (int j = 0; j < qk_per_thread; j++) {
        score += q[j] * k[j];
      }
      score = simd_sum(score);
      if (float_mask) {
        score += static_cast<U>(fmask[0]);
      }

      // Update the accumulators
      U new_max = max(max_score, score);
      U factor = fast::exp(max_score - new_max);
      U exp_score = fast::exp(score - new_max);

      max_score = new_max;
      sum_exp_score = sum_exp_score * factor + exp_score;

      // Update the output accumulator
      for (int j = 0; j < v_per_thread; j++) {
        o[j] = o[j] * factor + exp_score * values[j];
      }
    }

    // Move the pointers to the next kv
    keys += inner_k_stride;
    values += inner_v_stride;
    if (bool_mask) {
      bmask += BN * mask_kv_seq_stride;
    }
    if (float_mask) {
      fmask += BN * mask_kv_seq_stride;
    }
  }

  // Each thread has a partial part of the output so we need to combine them.

  // First let's communicate the max and sum_exp
  if (simd_lid == 0) {
    max_scores[simd_gid] = max_score;
    sum_exp_scores[simd_gid] = sum_exp_score;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  max_score = max_scores[simd_lid];
  U new_max = simd_max(max_score);
  U factor = fast::exp(max_score - new_max);
  sum_exp_score = simd_sum(sum_exp_scores[simd_lid] * factor);

  // Now we need to aggregate all the outputs
  for (int i = 0; i < v_per_thread; i++) {
    outputs[simd_lid * BD + simd_gid] = o[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    o[i] = simd_sum(outputs[simd_gid * BD + simd_lid] * factor);
    o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // And write the output
  if (simd_lid == 0) {
    for (int i = 0; i < v_per_thread; i++) {
      out[i] = static_cast<T>(o[i]);
    }
  }
}


template [[host_name("attn_small")]] [[kernel]]
decltype(sdpa_vector_small<bfloat, 256>) sdpa_vector_small<bfloat, 256>;
template [[host_name("attn_decode")]] [[kernel]]
decltype(sdpa_vector<bfloat, 256>) sdpa_vector<bfloat, 256>;
template [[host_name("attn_decode_split1")]] [[kernel]]
decltype(sdpa_vector_2pass_1<bfloat, 256>) sdpa_vector_2pass_1<bfloat, 256>;
template [[host_name("attn_decode_split2")]] [[kernel]]
decltype(sdpa_vector_2pass_2<bfloat, 256>) sdpa_vector_2pass_2<bfloat, 256>;

// Long-context decode reduction and output gate. SIMD lanes retain the
// parent block reduction order; BF16 boundaries match the separate kernels.
[[kernel]] void attn_decode_split2_gate(
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
