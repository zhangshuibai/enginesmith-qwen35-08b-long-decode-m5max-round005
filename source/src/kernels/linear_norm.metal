// QMV and RMS reductions adapted from MLX 0.32.4.
// Copyright (c) 2023-2025 Apple Inc. See third_party/linear_mlx/LICENSE.
// Fixed decode shape: M=1, K=1024, 256 threads. Each threadgroup repeats
// the parent's exact eight-SIMD RMS before its independent QMV rows.
// x, norm_out and y must be distinct: only gid.y=0 publishes norm_out.
#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;
struct LinearShape{uint M; uint N; uint K;};
kernel void qmv_mlp_g8_norm(
    device const uint* gate_w [[buffer(0)]],
    device const bfloat* gate_scales [[buffer(1)]],
    device const bfloat* gate_biases [[buffer(2)]],
    device const uint* up_w [[buffer(3)]],
    device const bfloat* up_scales [[buffer(4)]],
    device const bfloat* up_biases [[buffer(5)]],
    device const bfloat* x [[buffer(6)]],
    device bfloat* y [[buffer(7)]],
    device const bfloat* gain [[buffer(8)]],
    device bfloat* norm_out [[buffer(9)]],
    constant LinearShape& p [[buffer(10)]],
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
  threadgroup bfloat gate[16], up[16];
  device const uint* w = sg < 4 ? gate_w : up_w;
  device const bfloat* scales = sg < 4 ? gate_scales : up_scales;
  device const bfloat* biases = sg < 4 ? gate_biases : up_biases;
  uint row0 = gid.y * 16 + (sg % 4) * 4;
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
    if (lane == 0) {
      if (sg < 4) gate[(sg % 4)*4+r] = bfloat(value);
      else up[(sg % 4)*4+r] = bfloat(value);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < 16 && gid.y*16+tid < p.N && gid.x < p.M) {
    float g=float(gate[tid]);
    float e=float(bfloat(precise::exp(abs(g))));
    float denom=float(bfloat(1.0f+e));
    float prob=float(bfloat(1.0f/denom));
    bfloat s=bfloat(g<0 ? prob : 1.0f-prob);
    bfloat silu=bfloat(g*float(s));
    y[gid.x*p.N+gid.y*16+tid]=bfloat(float(silu)*float(up[tid]));
  }
}

