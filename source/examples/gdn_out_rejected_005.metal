// REJECTED: isolated gain did not transfer to full-model decode.
// Exact decode GDN gated norm followed by output QMV and residual addition.
// Parent expressions adapted from MLX 0.32.4 and mlx-lm 0.32.2.
// Copyright (c) 2023-2025 Apple Inc. See third_party/linear_mlx/LICENSE
// and GDN_INTERFACE.md.
//
// Fixed shape: M=1, N=1024, K=2048; 32 groups x 256 threads (eight SIMD
// groups). Every group repeats all sixteen independent 128-value head norms
// in private threadgroup memory, retaining four adjacent values per lane,
// each original SIMD reduction, and every BF16 rounding boundary.
// Only gid.y=0 publishes norm_out; it must not alias y or z. output/residual
// must not alias y, z or norm_out. output may alias residual because each row
// has one owner and no normalization prologue reads the residual.
#include <metal_stdlib>
using namespace metal;
struct LinearShape { uint M; uint N; uint K; };

kernel void gdn_out_g8(
    device const uint* w [[buffer(0)]],
    device const bfloat* scales [[buffer(1)]],
    device const bfloat* biases [[buffer(2)]],
    device const bfloat* y [[buffer(3)]],
    device const bfloat* z [[buffer(4)]],
    device const bfloat* weight [[buffer(5)]],
    device const bfloat* residual [[buffer(6)]],
    device bfloat* output [[buffer(7)]],
    device bfloat* norm_out [[buffer(8)]],
    constant LinearShape& p [[buffer(9)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup bfloat xn[2048];
  for (uint h = sg; h < 16; h += 8) {
    uint off = h * 128u + lane * 4u;
    float v[4];
    float ss = 0.0f;
    for (uint i = 0; i < 4; ++i) {
        v[i] = float(y[off + i]);
        ss += v[i] * v[i];
    }
    ss = simd_sum(ss);
    float inv = precise::rsqrt(ss / 128.0f + 1.0e-6f);
    for (uint i = 0; i < 4; ++i) {
        float norm = float(bfloat(float(bfloat(v[i] * inv)) * float(weight[lane * 4u + i])));
        float gate = float(z[off + i]);
        float sy = 1.0f / (1.0f + precise::exp(abs(gate)));
        float sig = gate < 0.0f ? sy : 1.0f - sy;
        xn[off + i] = bfloat((gate * sig) * norm);
    }
    if (gid.y == 0) for (uint i = 0; i < 4; ++i) norm_out[off+i] = xn[off+i];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint row0 = gid.y * 32 + sg * 4;
  float accum[4] = {0,0,0,0};
  for (uint k0 = 0; k0 < p.K; k0 += 512) {
    uint k = k0 + lane * 16;
    float xx[16];
    float xs = 0;
    for (uint j = 0; j < 16; j += 4) {
      float a = float(xn[gid.x * p.K + k + j]);
      float b = float(xn[gid.x * p.K + k + j + 1]);
      float c = float(xn[gid.x * p.K + k + j + 2]);
      float d = float(xn[gid.x * p.K + k + j + 3]);
      xs += a + b + c + d;
      xx[j] = a; xx[j+1] = b / 16.0f;
      xx[j+2] = c / 256.0f; xx[j+3] = d / 4096.0f;
    }
    for (uint r = 0; r < 4; ++r) {
      uint row = min(row0+r, p.N-1);
      device const ushort* ww = (device const ushort*)(w + row*(p.K/8)+k/8);
      float dotq = 0;
      for (uint j = 0; j < 4; ++j) {
        ushort v = ww[j];
        dotq += xx[4*j] * float(v & 0x000f)
              + xx[4*j+1] * float(v & 0x00f0)
              + xx[4*j+2] * float(v & 0x0f00)
              + xx[4*j+3] * float(v & 0xf000);
      }
      uint gi = row * (p.K/64) + k/64;
      accum[r] += float(scales[gi]) * dotq + xs * float(biases[gi]);
    }
  }
  for (uint r = 0; r < 4; ++r) {
    float value = simd_sum(accum[r]);
    if (lane == 0 && row0+r < p.N) output[gid.x*p.N+row0+r] = bfloat(float(residual[gid.x*p.N+row0+r])+float(bfloat(value)));
  }
}
