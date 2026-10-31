// Exact single-token GDN input RMS, projections and convolution/preparation.
// Parent expressions adapted from MLX 0.32.4 and mlx-lm 0.32.2.
// Copyright (c) 2023-2025 Apple Inc. See third_party/linear_mlx/LICENSE
// and GDN_INTERFACE.md.
//
// Fixed shape: M=1, K=1024, N={6144,2048,16,16}; 64 groups x 512 threads.
// Every group forms its own normalized input using the original eight-SIMD
// RMS reduction. The first 48 groups each own one 128-channel Q/K/V block;
// the last 16 own Z blocks. Sixteen SIMD groups compute two original four-row
// QMV walks each, retaining the projected BF16 values in raw[128]. A uniform
// barrier precedes the unchanged convolution and Q/K normalization in SIMD group 0.
// Each Q block's SIMD group 1 computes its own A/B head without reading another group.
// History is updated in place by its sole channel owner. norm_out, all four
// projections, conv, gates, x and history must be disjoint. Only group 0 writes
// norm_out. Multirow prefill uses the original separate kernels.
#include <metal_stdlib>
using namespace metal;
struct MultiLinearShape {uint M; uint N0; uint N1; uint N2; uint N3; uint K;};
// MLX's BF16 elementary operations round after each operation, including the
// exp and denominator inside sigmoid. These boundaries matter for this model.
inline float gdn_sigmoid_bf(float x) {
    float e = float(bfloat(precise::exp(abs(x))));
    float d = float(bfloat(1.0f + e));
    float y = float(bfloat(1.0f / d));
    return float(bfloat(x < 0.0f ? y : 1.0f - y));
}

// MLX utils.h log1p, specialized to x in [0,1] (the softplus exponential).
// Metal's standard library does not provide log1p.
inline float gdn_log1p(float x) {
    float xp1 = 1.0f + x;
    if (xp1 == 1.0f) return x;
    return x * (log(xp1) / (xp1 - 1.0f));
}

inline float gdn_softplus_bf(float x) {
    float e = float(bfloat(exp(-abs(x))));
    float l = float(bfloat(gdn_log1p(e)));
    return float(bfloat(max(x, 0.0f) + l));
}

inline bfloat4 project4(device const uint* w, device const bfloat* scales,
    device const bfloat* biases, threadgroup const bfloat* xn,
    uint row0, uint N, uint K, uint lane) {
  float accum[4] = {0,0,0,0};
  for (uint k0 = 0; k0 < K; k0 += 512) {
    uint k = k0 + lane * 16;
    float xx[16];
    float xs = 0;
    for (uint j = 0; j < 16; j += 4) {
      float a = float(xn[k + j]);
      float b = float(xn[k + j + 1]);
      float c = float(xn[k + j + 2]);
      float d = float(xn[k + j + 3]);
      xs += a + b + c + d;
      xx[j] = a; xx[j+1] = b / 16.0f;
      xx[j+2] = c / 256.0f; xx[j+3] = d / 4096.0f;
    }
    for (uint r = 0; r < 4; ++r) {
      uint row = min(row0+r, N-1);
      device const ushort* ww = (device const ushort*)(w + row*(K/8)+k/8);
      float dotq = 0;
      for (uint j = 0; j < 4; ++j) {
        ushort v = ww[j];
        dotq += xx[4*j] * float(v & 0x000f)
              + xx[4*j+1] * float(v & 0x00f0)
              + xx[4*j+2] * float(v & 0x0f00)
              + xx[4*j+3] * float(v & 0xf000);
      }
      uint gi = row * (K/64) + k/64;
      accum[r] += float(scales[gi]) * dotq + xs * float(biases[gi]);
    }
  }
  return bfloat4(bfloat(simd_sum(accum[0])), bfloat(simd_sum(accum[1])),
                 bfloat(simd_sum(accum[2])), bfloat(simd_sum(accum[3])));
}
kernel void gdn_input_g16(
    device const uint* w0 [[buffer(0)]],
    device const bfloat* s0 [[buffer(1)]],
    device const bfloat* b0 [[buffer(2)]],
    device bfloat* y0 [[buffer(3)]],
    device const uint* w1 [[buffer(4)]],
    device const bfloat* s1 [[buffer(5)]],
    device const bfloat* b1 [[buffer(6)]],
    device bfloat* y1 [[buffer(7)]],
    device const uint* w2 [[buffer(8)]],
    device const bfloat* s2 [[buffer(9)]],
    device const bfloat* b2 [[buffer(10)]],
    device bfloat* y2 [[buffer(11)]],
    device const uint* w3 [[buffer(12)]],
    device const bfloat* s3 [[buffer(13)]],
    device const bfloat* b3 [[buffer(14)]],
    device bfloat* y3 [[buffer(15)]],
    device const bfloat* x [[buffer(16)]],
    device const bfloat* gain [[buffer(17)]],
    device bfloat* norm_out [[buffer(18)]],
    device bfloat* history [[buffer(19)]],
    device const bfloat* conv_weight [[buffer(20)]],
    device bfloat* conv [[buffer(21)]],
    device const float* a_log [[buffer(22)]],
    device const bfloat* dt_bias [[buffer(23)]],
    device float* g [[buffer(24)]],
    device bfloat* beta [[buffer(25)]],
    constant MultiLinearShape& shape [[buffer(26)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup bfloat xn[1024];
  threadgroup float sums[32];
  threadgroup float inv;
  float v[4], acc=0;
  for (uint j=0;j<4;++j) {
    uint c=tid*4+j;
    v[j]=c<1024 ? float(x[c]) : 0;
    acc += v[j]*v[j];
  }
  acc=simd_sum(acc);
  if (sg==0) sums[lane]=0;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane==0 && sg<8) sums[sg]=acc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg==0) {
    acc=simd_sum(sums[lane]);
    if (lane==0) inv=precise::rsqrt(acc/float(1024)+1e-6f);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint j=0;j<4;++j) {
    uint c=tid*4+j;
    if (c<1024) {
      xn[c]=bfloat(float(bfloat(v[j]*inv))*float(gain[c]));
      if (gid.y==0) norm_out[c]=xn[c];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  threadgroup bfloat raw[128];
  uint group = gid.y;
  bool qkv = group < 48;
  uint block_group = qkv ? group : group - 48;
  device const uint* w = qkv ? w0 : w1;
  device const bfloat* scales = qkv ? s0 : s1;
  device const bfloat* biases = qkv ? b0 : b1;
  device bfloat* output = qkv ? y0 : y1;
  uint N = qkv ? shape.N0 : shape.N1;
  for (uint sub = 0; sub < 2; ++sub) {
    uint row = block_group * 128 + (sub * 16 + sg) * 4;
    bfloat4 value = project4(w, scales, biases, xn, row, N, shape.K, lane);
    if (lane == 0) {
      for (uint r = 0; r < 4; ++r) {
        output[row+r] = value[r];
        raw[(sub * 16 + sg) * 4 + r] = value[r];
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (qkv && sg == 0) {
    uint block = group / 16u;
    uint h = group % 16u;
    uint off = block * 2048u + h * 128u + lane * 4u;
    float v[4];
    float ss = 0.0f;
    for (uint i = 0; i < 4; ++i) {
        uint c = off + i;
        bfloat old[3] = { history[c], history[6144u + c], history[12288u + c] };
        float acc = 0.0f;
        for (uint tap = 0; tap < 4; ++tap) {
            float x = tap < 3 ? float(old[tap]) : float(raw[lane * 4u + i]);
            acc = fma(x, float(conv_weight[c * 4u + tap]), acc);
        }
        float x = float(bfloat(acc));
        v[i] = float(bfloat(x * gdn_sigmoid_bf(x)));
        ss += v[i] * v[i];
        history[c] = old[1];
        history[6144u + c] = old[2];
        history[12288u + c] = raw[lane * 4u + i];
    }
    if (block < 2u) {
        ss = simd_sum(ss);
        float inv = precise::rsqrt(ss / 128.0f + 1.0e-6f / 128.0f);
        const float q_scale = 0.0078125f;
        const float k_scale = float(bfloat(0.08838834764831845f));
        float scale = block == 0u ? q_scale : k_scale;
        for (uint i = 0; i < 4; ++i) {
            conv[off + i] = bfloat(float(bfloat(v[i] * inv)) * scale);
        }
    } else {
        for (uint i = 0; i < 4; ++i) conv[off + i] = bfloat(v[i]);
    }
  }
  // The Q block's second SIMD group owns this head's scalar projections.
  // All four original rows are computed, then the matching BF16 row selected.
  // No group reads another group's raw projection or history.
  if (group < 16 && sg == 1) {
    uint row0 = (group / 4) * 4;
    bfloat4 a4 = project4(w2, s2, b2, xn, row0, shape.N2, shape.K, lane);
    bfloat4 b4 = project4(w3, s3, b3, xn, row0, shape.N3, shape.K, lane);
    if (lane == 0) {
      bfloat a = a4[group % 4], b = b4[group % 4];
      y2[group] = a;
      y3[group] = b;
      float ax = float(bfloat(float(a) + float(dt_bias[group])));
      g[group] = precise::exp(-precise::exp(a_log[group]) * gdn_softplus_bf(ax));
      beta[group] = bfloat(gdn_sigmoid_bf(float(b)));
    }
  }
}
