// QMV and logsumexp reductions adapted from MLX 0.32.4.
// Copyright (c) 2023-2025 Apple Inc. See third_party/linear_mlx/LICENSE.
// Fixed head shape: M=1, K=1024, N=248320. Each SIMD group owns one
// original greedy thread's four-logit walk, including the BF16 boundary.
// The two 1024-F32 partial buffers and logits must not overlap.
#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;
struct LinearShape{uint M;uint N;uint K;};

kernel void head_lse_g8(
 device const uint* w [[buffer(0)]],device const bfloat* scales [[buffer(1)]],device const bfloat* biases [[buffer(2)]],
 device const bfloat* x [[buffer(3)]],device bfloat* y [[buffer(4)]],device float* partial_max [[buffer(5)]],device float* partial_sum [[buffer(6)]],constant LinearShape& p [[buffer(7)]],
 uint gid [[threadgroup_position_in_grid]],uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]){
 uint owner=gid*8+sg;
 float vmax=-MAXFLOAT,normalizer=0;
 for(uint tile=0;tile<(p.N+4095)/4096;++tile){
  uint row0=tile*4096+owner*4;
  float vals[4]={-INFINITY,-INFINITY,-INFINITY,-INFINITY};
  if(row0<p.N){
  float accum[4] = {0,0,0,0};
  for (uint k0 = 0; k0 < p.K; k0 += 512) {
    uint k = k0 + lane * 16;
    float xx[16];
    float xs = 0;
    for (uint j = 0; j < 16; j += 4) {
      float a = float(x[k + j]);
      float b = float(x[k + j + 1]);
      float c = float(x[k + j + 2]);
      float d = float(x[k + j + 3]);
      xs += a + b + c + d;
      xx[j] = a; xx[j+1] = b / 16.0f;
      xx[j+2] = c / 256.0f; xx[j+3] = d / 4096.0f;
    }
    for (uint qr = 0; qr < 4; ++qr) {
      uint row = min(row0+qr, p.N-1);
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
      accum[qr] += float(scales[gi]) * dotq + xs * float(biases[gi]);
    }
  }

   for(uint qr=0;qr<4;++qr){
    float value=simd_sum(accum[qr]);
    if(lane==0&&row0+qr<p.N){y[row0+qr]=bfloat(value);vals[qr]=float(bfloat(value));}
   }
  }
  if(lane==0){
   float prev=vmax;
   for(uint j=0;j<4;++j)vmax=max(vmax,vals[j]);
   normalizer *= fast::exp(prev-vmax);
   for(uint j=0;j<4;++j)normalizer+=fast::exp(vals[j]-vmax);
  }
 }
 if(lane==0){partial_max[owner]=vmax;partial_sum[owner]=normalizer;}
}

kernel void head_lse_tail(device const bfloat* logits [[buffer(0)]],device const float* partial_max [[buffer(1)]],device const float* partial_sum [[buffer(2)]],device uint* token [[buffer(3)]],constant uint& count [[buffer(4)]], uint tid [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]){
 threadgroup float maxima[32],sums[32];threadgroup uint ids[32];threadgroup bfloat lse;
 float vmax=partial_max[tid],normalizer=partial_sum[tid];
  float prev=vmax;
  vmax=simd_max(vmax);
  normalizer*=fast::exp(prev-vmax);
  normalizer=simd_sum(normalizer);
  prev=vmax;
  if(lane==0) maxima[sg]=vmax;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  vmax=simd_max(maxima[lane]);
  normalizer*=fast::exp(prev-vmax);
  if(lane==0) sums[sg]=normalizer;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  normalizer=simd_sum(sums[lane]);
  if(tid==0) lse=bfloat(log(normalizer)+vmax);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float best=-INFINITY;
  uint id=0xffffffff;
  for(uint i=tid;i<count;i+=1024) {
    float v=float(bfloat(float(logits[i])-float(lse)));
    if(v>best || (v==best && i<id)) {best=v; id=i;}
  }
  best=simd_max(best);
  id=simd_min(float(bfloat(float(logits[min(id,count-1)])-float(lse)))==best ? id : 0xffffffff);
  if(lane==0) {maxima[sg]=best; ids[sg]=id;}
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if(sg==0) {
    best=simd_max(maxima[lane]);
    id=simd_min(maxima[lane]==best ? ids[lane] : 0xffffffff);
    if(lane==0) token[0]=id;
  }
}
