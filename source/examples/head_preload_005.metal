// QMV and logsumexp reductions adapted from MLX 0.32.4.
// Copyright (c) 2023-2025 Apple Inc. See third_party/linear_mlx/LICENSE.
// Fixed head shape: M=1, K=1024, N=248320. Each SIMD group owns one
// original greedy thread's four-logit walk, including the BF16 boundary.
// The two 1024-F32 partial buffers and logits must not overlap.
#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;
struct LinearShape{uint M;uint N;uint K;};

kernel void head_lse_g8_preload(
 device const uint* w [[buffer(0)]],device const bfloat* scales [[buffer(1)]],device const bfloat* biases [[buffer(2)]],
 device const bfloat* x [[buffer(3)]],device bfloat* y [[buffer(4)]],device float* partial_max [[buffer(5)]],device float* partial_sum [[buffer(6)]],constant LinearShape& p [[buffer(7)]],
 uint gid [[threadgroup_position_in_grid]],uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]){
 uint owner=gid*8+sg;
 float pre_x[32],pre_sum[2];
 for(uint k0=0;k0<p.K;k0+=512){
   uint part=k0/512;
   uint k=k0+lane*16;
    
    float xs = 0;
    for (uint j = 0; j < 16; j += 4) {
      float a = float(x[k + j]);
      float b = float(x[k + j + 1]);
      float c = float(x[k + j + 2]);
      float d = float(x[k + j + 3]);
      xs += a + b + c + d;
      pre_x[part*16+j] = a; pre_x[part*16+j+1] = b / 16.0f;
      pre_x[part*16+j+2] = c / 256.0f; pre_x[part*16+j+3] = d / 4096.0f;
    }
    pre_sum[part]=xs;
 }
 float vmax=-MAXFLOAT,normalizer=0;
 for(uint tile=0;tile<(p.N+4095)/4096;++tile){
  uint row0=tile*4096+owner*4;
  float vals[4]={-INFINITY,-INFINITY,-INFINITY,-INFINITY};
  if(row0<p.N){
  float accum[4] = {0,0,0,0};
  for (uint k0 = 0; k0 < p.K; k0 += 512) {
    uint k = k0 + lane * 16;
    float xs=pre_sum[k0/512];
    for (uint qr = 0; qr < 4; ++qr) {
      uint row = min(row0+qr, p.N-1);
      device const ushort* ww = (device const ushort*)(w + row*(p.K/8)+k/8);
      float dotq = 0;
      for (uint j = 0; j < 4; ++j) {
        ushort v = ww[j];
        dotq += pre_x[(k0/512)*16+4*j] * float(v & 0x000f)
              + pre_x[(k0/512)*16+4*j+1] * float(v & 0x00f0)
              + pre_x[(k0/512)*16+4*j+2] * float(v & 0x0f00)
              + pre_x[(k0/512)*16+4*j+3] * float(v & 0xf000);
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


kernel void head_lse_g8_vector(
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
      ushort4 packed = *((device const ushort4*)ww);
      float dotq = 0;
      for (uint j = 0; j < 4; ++j) {
        ushort v = packed[j];
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


kernel void head_lse_g8_preload_vector(
 device const uint* w [[buffer(0)]],device const bfloat* scales [[buffer(1)]],device const bfloat* biases [[buffer(2)]],
 device const bfloat* x [[buffer(3)]],device bfloat* y [[buffer(4)]],device float* partial_max [[buffer(5)]],device float* partial_sum [[buffer(6)]],constant LinearShape& p [[buffer(7)]],
 uint gid [[threadgroup_position_in_grid]],uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]){
 uint owner=gid*8+sg;
 float pre_x[32],pre_sum[2];
 for(uint k0=0;k0<p.K;k0+=512){
   uint part=k0/512;
   uint k=k0+lane*16;
    
    float xs = 0;
    for (uint j = 0; j < 16; j += 4) {
      float a = float(x[k + j]);
      float b = float(x[k + j + 1]);
      float c = float(x[k + j + 2]);
      float d = float(x[k + j + 3]);
      xs += a + b + c + d;
      pre_x[part*16+j] = a; pre_x[part*16+j+1] = b / 16.0f;
      pre_x[part*16+j+2] = c / 256.0f; pre_x[part*16+j+3] = d / 4096.0f;
    }
    pre_sum[part]=xs;
 }
 float vmax=-MAXFLOAT,normalizer=0;
 for(uint tile=0;tile<(p.N+4095)/4096;++tile){
  uint row0=tile*4096+owner*4;
  float vals[4]={-INFINITY,-INFINITY,-INFINITY,-INFINITY};
  if(row0<p.N){
  float accum[4] = {0,0,0,0};
  for (uint k0 = 0; k0 < p.K; k0 += 512) {
    uint k = k0 + lane * 16;
    float xs=pre_sum[k0/512];
    for (uint qr = 0; qr < 4; ++qr) {
      uint row = min(row0+qr, p.N-1);
      device const ushort* ww = (device const ushort*)(w + row*(p.K/8)+k/8);
      ushort4 packed = *((device const ushort4*)ww);
      float dotq = 0;
      for (uint j = 0; j < 4; ++j) {
        ushort v = packed[j];
        dotq += pre_x[(k0/512)*16+4*j] * float(v & 0x000f)
              + pre_x[(k0/512)*16+4*j+1] * float(v & 0x00f0)
              + pre_x[(k0/512)*16+4*j+2] * float(v & 0x0f00)
              + pre_x[(k0/512)*16+4*j+3] * float(v & 0xf000);
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

