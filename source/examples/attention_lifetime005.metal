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

template <typename T, int D, int V = D, int TK = 4>
[[kernel]] void scores_then_v2(
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
    U scores[TK];
    for (int t = 0; t < TK; t++) {
      U score = 0;
      for (int j = 0; j < qk_per_thread; j++) {
        score += q[j] * static_cast<U>(k_reg[t][j]);
      }
      scores[t] = simd_sum(score);
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
        U score = scores[t];
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


template [[host_name("scores_then_v2")]] [[kernel]]
decltype(scores_then_v2<bfloat,256,256,2>) scores_then_v2<bfloat,256,256,2>;

template <typename T, int D, int V = D, int TK = 4>
[[kernel]] void scores_then_v4(
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
    U scores[TK];
    for (int t = 0; t < TK; t++) {
      U score = 0;
      for (int j = 0; j < qk_per_thread; j++) {
        score += q[j] * static_cast<U>(k_reg[t][j]);
      }
      scores[t] = simd_sum(score);
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
        U score = scores[t];
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


template [[host_name("scores_then_v4")]] [[kernel]]
decltype(scores_then_v4<bfloat,256,256,4>) scores_then_v4<bfloat,256,256,4>;

template <typename T, int D, int V = D, int TK = 4>
[[kernel]] void scores_then_v8(
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
    U scores[TK];
    for (int t = 0; t < TK; t++) {
      U score = 0;
      for (int j = 0; j < qk_per_thread; j++) {
        score += q[j] * static_cast<U>(k_reg[t][j]);
      }
      scores[t] = simd_sum(score);
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
        U score = scores[t];
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


template [[host_name("scores_then_v8")]] [[kernel]]
decltype(scores_then_v8<bfloat,256,256,8>) scores_then_v8<bfloat,256,256,8>;

template <typename T, int D, int V = D, int TK = 4>
[[kernel]] void all_loads_then_scores2(
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

    U scores[TK];
    for (int t = 0; t < TK; t++) {
      U score = 0;
      for (int j = 0; j < qk_per_thread; j++) {
        score += q[j] * static_cast<U>(k_reg[t][j]);
      }
      scores[t] = simd_sum(score);
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
        U score = scores[t];
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


template [[host_name("all_loads_then_scores2")]] [[kernel]]
decltype(all_loads_then_scores2<bfloat,256,256,2>) all_loads_then_scores2<bfloat,256,256,2>;

template <typename T, int D, int V = D, int TK = 4>
[[kernel]] void all_loads_then_scores4(
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

    U scores[TK];
    for (int t = 0; t < TK; t++) {
      U score = 0;
      for (int j = 0; j < qk_per_thread; j++) {
        score += q[j] * static_cast<U>(k_reg[t][j]);
      }
      scores[t] = simd_sum(score);
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
        U score = scores[t];
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


template [[host_name("all_loads_then_scores4")]] [[kernel]]
decltype(all_loads_then_scores4<bfloat,256,256,4>) all_loads_then_scores4<bfloat,256,256,4>;

template <typename T, int D, int V = D, int TK = 4>
[[kernel]] void all_loads_then_scores8(
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

    U scores[TK];
    for (int t = 0; t < TK; t++) {
      U score = 0;
      for (int j = 0; j < qk_per_thread; j++) {
        score += q[j] * static_cast<U>(k_reg[t][j]);
      }
      scores[t] = simd_sum(score);
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
        U score = scores[t];
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


template [[host_name("all_loads_then_scores8")]] [[kernel]]
decltype(all_loads_then_scores8<bfloat,256,256,8>) all_loads_then_scores8<bfloat,256,256,8>;

template <typename T, int D, int V = D, int TK = 4>
[[kernel]] void packed4(
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
    uint4 k_reg[TK];
    uint4 v_reg[TK];
    // Each lane owns eight contiguous BF16 elements, aligned to 16 bytes.
    // Explicit vector loads retain every value and arithmetic order while
    // expressing each K/V row as one 16-byte load per lane.
    for (int t = 0; t < TK; t++) {
      k_reg[t] = reinterpret_cast<const device uint4*>(keys + t * stride_k_tok)[0];
      v_reg[t] = reinterpret_cast<const device uint4*>(values + t * stride_v_tok)[0];
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
        vec<T, 4> kl = as_type<vec<T, 4>>(k_reg[t].xy);
        vec<T, 4> kh = as_type<vec<T, 4>>(k_reg[t].zw);
        vec<T, 4> vl = as_type<vec<T, 4>>(v_reg[t].xy);
        vec<T, 4> vh = as_type<vec<T, 4>>(v_reg[t].zw);
        U score = 0;
        for (int j = 0; j < qk_per_thread; j++) {
          score += q[j] * static_cast<U>(j < 4 ? kl[j] : kh[j - 4]);
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
          o[j] = o[j] * factor + exp_score * static_cast<U>(j < 4 ? vl[j] : vh[j - 4]);
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


template [[host_name("packed4")]] [[kernel]]
decltype(packed4<bfloat,256,256,4>) packed4<bfloat,256,256,4>;

template <typename T, int D, int V = D, int TK = 4>
[[kernel]] void packed8(
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
    uint4 k_reg[TK];
    uint4 v_reg[TK];
    // Each lane owns eight contiguous BF16 elements, aligned to 16 bytes.
    // Explicit vector loads retain every value and arithmetic order while
    // expressing each K/V row as one 16-byte load per lane.
    for (int t = 0; t < TK; t++) {
      k_reg[t] = reinterpret_cast<const device uint4*>(keys + t * stride_k_tok)[0];
      v_reg[t] = reinterpret_cast<const device uint4*>(values + t * stride_v_tok)[0];
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
        vec<T, 4> kl = as_type<vec<T, 4>>(k_reg[t].xy);
        vec<T, 4> kh = as_type<vec<T, 4>>(k_reg[t].zw);
        vec<T, 4> vl = as_type<vec<T, 4>>(v_reg[t].xy);
        vec<T, 4> vh = as_type<vec<T, 4>>(v_reg[t].zw);
        U score = 0;
        for (int j = 0; j < qk_per_thread; j++) {
          score += q[j] * static_cast<U>(j < 4 ? kl[j] : kh[j - 4]);
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
          o[j] = o[j] * factor + exp_score * static_cast<U>(j < 4 ? vl[j] : vh[j - 4]);
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


template [[host_name("packed8")]] [[kernel]]
decltype(packed8<bfloat,256,256,8>) packed8<bfloat,256,256,8>;

template <typename T, int D, int V = D, int TK = 4>
[[kernel]] void paired_exp4(
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
        U exp_arg = (simd_lid & 1) == 0 ? (max_score - new_max) : (score - new_max);
        U exp_pair = fast::exp(exp_arg);
        U factor = simd_broadcast(exp_pair, 0);
        U exp_score = simd_broadcast(exp_pair, 1);
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


template [[host_name("paired_exp4")]] [[kernel]]
decltype(paired_exp4<bfloat,256,256,4>) paired_exp4<bfloat,256,256,4>;

template <typename T, int D, int V = D, int TK = 4>
[[kernel]] void batch_exp4(
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
    U scores[TK];
    for (int t = 0; t < TK; t++) {
      U score = 0;
      for (int j = 0; j < qk_per_thread; j++) {
        score += q[j] * static_cast<U>(k_reg[t][j]);
      }
      scores[t] = simd_sum(score);
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

    U exp_a[TK], exp_b[TK];
    for (int t = 0; t < TK; t++) {
      U new_max = max(max_score, scores[t]);
      exp_a[t] = max_score - new_max;
      exp_b[t] = scores[t] - new_max;
      max_score = new_max;
    }
    U exp_arg = exp_a[0];
    for (int t = 0; t < TK; t++) {
      exp_arg = simd_lid == uint(2*t) ? exp_a[t] : exp_arg;
      exp_arg = simd_lid == uint(2*t+1) ? exp_b[t] : exp_arg;
    }
    U exp_batch = fast::exp(exp_arg);
    for (int t = 0; t < TK; t++) {
      U factor = simd_broadcast(exp_batch, 2*t);
      U exp_score = simd_broadcast(exp_batch, 2*t+1);
      sum_exp_score = sum_exp_score * factor + exp_score;
      for (int j = 0; j < v_per_thread; j++) {
        o[j] = o[j] * factor + exp_score * static_cast<U>(v_reg[t][j]);
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


template [[host_name("batch_exp4")]] [[kernel]]
decltype(batch_exp4<bfloat,256,256,4>) batch_exp4<bfloat,256,256,4>;
