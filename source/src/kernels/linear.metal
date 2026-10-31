// EngineSmith Qwen3.5 affine Q4 baseline kernels.
// The QMV, RMS, and logsumexp reductions are adapted from MLX 0.32.4.
// Copyright (c) 2023-2025 Apple Inc. See third_party/linear_mlx/LICENSE.
#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

struct LinearShape { uint M; uint N; uint K; };
struct NormShape { uint rows; uint cols; float eps; };

kernel void embedding_q4(
    device const uint* w [[buffer(0)]],
    device const bfloat* scales [[buffer(1)]],
    device const bfloat* biases [[buffer(2)]],
    device const uint* tokens [[buffer(3)]],
    device bfloat* out [[buffer(4)]],
    constant LinearShape& p [[buffer(5)]],
    uint i [[thread_position_in_grid]]) {
  if (i >= p.M * p.K) return;
  uint row = tokens[i / p.K], k = i % p.K;
  uint q = (w[row * (p.K / 8) + k / 8] >> (4 * (k % 8))) & 15;
  uint g = row * (p.K / 64) + k / 64;
  out[i] = bfloat(float(scales[g]) * float(q) + float(biases[g]));
}

// Same 16-value/lane affine decomposition as MLX qmv_fast_impl<bf16,64,4>.
// One threadgroup = two SIMD groups, four rows per SIMD group.
kernel void qmv_q4(
    device const uint* w [[buffer(0)]],
    device const bfloat* scales [[buffer(1)]],
    device const bfloat* biases [[buffer(2)]],
    device const bfloat* x [[buffer(3)]],
    device bfloat* y [[buffer(4)]],
    constant LinearShape& p [[buffer(5)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  uint row0 = gid.y * 8 + sg * 4;
  if (row0 >= p.N || gid.x >= p.M) return;
  float accum[4] = {0,0,0,0};
  for (uint k0 = 0; k0 < p.K; k0 += 512) {
    uint k = k0 + lane * 16;
    float xx[16];
    float xs = 0;
    for (uint j = 0; j < 16; j += 4) {
      float a = float(x[gid.x * p.K + k + j]);
      float b = float(x[gid.x * p.K + k + j + 1]);
      float c = float(x[gid.x * p.K + k + j + 2]);
      float d = float(x[gid.x * p.K + k + j + 3]);
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
    if (lane == 0 && row0+r < p.N) y[gid.x*p.N+row0+r] = bfloat(value);
  }
}

// Residual add epilogue retains the projected BF16 boundary. The residual
// output may alias residual, but must not alias the QMV input x.
kernel void qmv_add_q4(
    device const uint* w [[buffer(0)]],
    device const bfloat* scales [[buffer(1)]],
    device const bfloat* biases [[buffer(2)]],
    device const bfloat* x [[buffer(3)]],
    device const bfloat* residual [[buffer(4)]],
    device bfloat* y [[buffer(5)]],
    constant LinearShape& p [[buffer(6)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  uint row0 = gid.y * 8 + sg * 4;
  if (row0 >= p.N || gid.x >= p.M) return;
  float accum[4] = {0,0,0,0};
  for (uint k0 = 0; k0 < p.K; k0 += 512) {
    uint k = k0 + lane * 16;
    float xx[16];
    float xs = 0;
    for (uint j = 0; j < 16; j += 4) {
      float a = float(x[gid.x * p.K + k + j]);
      float b = float(x[gid.x * p.K + k + j + 1]);
      float c = float(x[gid.x * p.K + k + j + 2]);
      float d = float(x[gid.x * p.K + k + j + 3]);
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
    if (lane == 0 && row0+r < p.N) y[gid.x*p.N+row0+r] = bfloat(float(residual[gid.x*p.N+row0+r])+float(bfloat(value)));
  }
}

// Shared-input QMV projections retain each matrix's original row index,
// lane walk, scale indexing and SIMD reduction. N3=0 supports three matrices.
struct MultiLinearShape { uint M; uint N0; uint N1; uint N2; uint N3; uint K; };
kernel void qmv_multi_q4(
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
    constant MultiLinearShape& shape [[buffer(17)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  uint group=gid.y;
  uint n=shape.N0;
  device const uint* w=w0;
  device const bfloat* scales=s0;
  device const bfloat* biases=b0;
  device bfloat* y=y0;
  if (group >= (n+7)/8) {
    group -= (n+7)/8;
    n=shape.N1;w=w1;scales=s1;biases=b1;y=y1;
    if (group >= (n+7)/8) {
      group -= (n+7)/8;
      n=shape.N2;w=w2;scales=s2;biases=b2;y=y2;
      if (group >= (n+7)/8) {
        group -= (n+7)/8;
        n=shape.N3;w=w3;scales=s3;biases=b3;y=y3;
      }
    }
  }
  const LinearShape p={shape.M,n,shape.K};
  uint row0 = group * 8 + sg * 4;
  if (row0 >= p.N || gid.x >= p.M) return;
  float accum[4] = {0,0,0,0};
  for (uint k0 = 0; k0 < p.K; k0 += 512) {
    uint k = k0 + lane * 16;
    float xx[16];
    float xs = 0;
    for (uint j = 0; j < 16; j += 4) {
      float a = float(x[gid.x * p.K + k + j]);
      float b = float(x[gid.x * p.K + k + j + 1]);
      float c = float(x[gid.x * p.K + k + j + 2]);
      float d = float(x[gid.x * p.K + k + j + 3]);
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
    if (lane == 0 && row0+r < p.N) y[gid.x*p.N+row0+r] = bfloat(value);
  }
}

// Decode gate/up projections share a dispatch, preserving each projection's
// four-row SIMD walk. Rounded producer values cross the original BF16 boundary
// in threadgroup memory before the unchanged SwiGLU expression.
kernel void qmv_gate_up_swiglu_q4(
    device const uint* gate_w [[buffer(0)]],
    device const bfloat* gate_scales [[buffer(1)]],
    device const bfloat* gate_biases [[buffer(2)]],
    device const uint* up_w [[buffer(3)]],
    device const bfloat* up_scales [[buffer(4)]],
    device const bfloat* up_biases [[buffer(5)]],
    device const bfloat* x [[buffer(6)]],
    device bfloat* y [[buffer(7)]],
    constant LinearShape& p [[buffer(8)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup bfloat gate[8], up[8];
  device const uint* w = sg < 2 ? gate_w : up_w;
  device const bfloat* scales = sg < 2 ? gate_scales : up_scales;
  device const bfloat* biases = sg < 2 ? gate_biases : up_biases;
  uint row0 = gid.y * 8 + (sg % 2) * 4;
  float accum[4] = {0,0,0,0};
  for (uint k0 = 0; k0 < p.K; k0 += 512) {
    uint k = k0 + lane * 16;
    float xx[16];
    float xs = 0;
    for (uint j = 0; j < 16; j += 4) {
      float a = float(x[gid.x * p.K + k + j]);
      float b = float(x[gid.x * p.K + k + j + 1]);
      float c = float(x[gid.x * p.K + k + j + 2]);
      float d = float(x[gid.x * p.K + k + j + 3]);
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
      if (sg < 2) gate[(sg % 2)*4+r] = bfloat(value);
      else up[(sg % 2)*4+r] = bfloat(value);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < 8 && gid.y*8+tid < p.N && gid.x < p.M) {
    float g=float(gate[tid]);
    float e=float(bfloat(precise::exp(abs(g))));
    float denom=float(bfloat(1.0f+e));
    float prob=float(bfloat(1.0f/denom));
    bfloat s=bfloat(g<0 ? prob : 1.0f-prob);
    bfloat silu=bfloat(g*float(s));
    y[gid.x*p.N+gid.y*8+tid]=bfloat(float(silu)*float(up[tid]));
  }
}

// Plain BF16-dequant matrix-product comparison route, not used for M5 production.
kernel void qmm_q4_plain(
    device const uint* w [[buffer(0)]],
    device const bfloat* scales [[buffer(1)]],
    device const bfloat* biases [[buffer(2)]],
    device const bfloat* x [[buffer(3)]],
    device bfloat* y [[buffer(4)]],
    constant LinearShape& p [[buffer(5)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  uint row=gid.x, n=gid.y;
  if (row >= p.M || n >= p.N) return;
  float v=0;
  for (uint k=lane; k<p.K; k+=32) {
    uint q=(w[n*(p.K/8)+k/8] >> (4*(k%8)))&15;
    uint g=n*(p.K/64)+k/64;
    bfloat wt=bfloat(float(scales[g])*float(q)+float(biases[g]));
    v += float(x[row*p.K+k])*float(wt);
  }
  v=simd_sum(v);
  if (lane==0) y[row*p.N+n]=bfloat(v);
}

// Four contiguous values/thread, same two-level reduction as MLX rms_single_row.
kernel void rms_norm_bf16(
    device const bfloat* x [[buffer(0)]],
    device const bfloat* gain [[buffer(1)]],
    device bfloat* out [[buffer(2)]],
    constant NormShape& p [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float sums[32];
  threadgroup float inv;
  float v[4], acc=0;
  for (uint j=0;j<4;++j) {
    uint c=tid*4+j;
    v[j]=c<p.cols ? float(x[row*p.cols+c]) : 0;
    acc += v[j]*v[j];
  }
  acc=simd_sum(acc);
  if (sg==0) sums[lane]=0;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane==0) sums[sg]=acc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg==0) {
    acc=simd_sum(sums[lane]);
    if (lane==0) inv=precise::rsqrt(acc/float(p.cols)+p.eps);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint j=0;j<4;++j) {
    uint c=tid*4+j;
    if (c<p.cols) out[row*p.cols+c]=bfloat(float(bfloat(v[j]*inv))*float(gain[c]));
  }
}

// The residual's BF16 store is retained before the RMS statistics. Each thread
// owns its four residual elements, so residual_out may alias either add input.
kernel void add_rms_norm_bf16(
    device const bfloat* a [[buffer(0)]],
    device const bfloat* b [[buffer(1)]],
    device const bfloat* gain [[buffer(2)]],
    device bfloat* residual_out [[buffer(3)]],
    device bfloat* out [[buffer(4)]],
    constant NormShape& p [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float sums[32];
  threadgroup float inv;
  float v[4], acc=0;
  for (uint j=0;j<4;++j) {
    uint c=tid*4+j;
    bfloat sum=c<p.cols ? bfloat(float(a[row*p.cols+c])+float(b[row*p.cols+c])) : bfloat(0);
    if (c<p.cols) residual_out[row*p.cols+c]=sum;
    v[j]=float(sum);
    acc += v[j]*v[j];
  }
  acc=simd_sum(acc);
  if (sg==0) sums[lane]=0;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane==0) sums[sg]=acc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg==0) {
    acc=simd_sum(sums[lane]);
    if (lane==0) inv=precise::rsqrt(acc/float(p.cols)+p.eps);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint j=0;j<4;++j) {
    uint c=tid*4+j;
    if (c<p.cols) out[row*p.cols+c]=bfloat(float(bfloat(v[j]*inv))*float(gain[c]));
  }
}

kernel void add_bf16(
    device const bfloat* a [[buffer(0)]],
    device const bfloat* b [[buffer(1)]],
    device bfloat* out [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint i [[thread_position_in_grid]]) {
  if (i<count) out[i]=bfloat(float(a[i])+float(b[i]));
}

// The compiled BF16 graph retains a BF16 boundary at each expression node.
kernel void swiglu_bf16(
    device const bfloat* gate [[buffer(0)]],
    device const bfloat* up [[buffer(1)]],
    device bfloat* out [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint i [[thread_position_in_grid]]) {
  if (i>=count) return;
  float g=float(gate[i]);
  float e=float(bfloat(precise::exp(abs(g))));
  float denom=float(bfloat(1.0f+e));
  float prob=float(bfloat(1.0f/denom));
  bfloat s=bfloat(g<0 ? prob : 1.0f-prob);
  bfloat silu=bfloat(g*float(s));
  out[i]=bfloat(float(silu)*float(up[i]));
}

// Single 1024-thread group. Compute BF16 logsumexp then subtract in BF16,
// preserving the extra ties introduced by low-precision log probabilities.
kernel void greedy_bf16(
    device const bfloat* logits [[buffer(0)]],
    device uint* token [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float maxima[32], sums[32];
  threadgroup uint ids[32];
  threadgroup bfloat lse;
  float vmax=-MAXFLOAT, normalizer=0;
  for(uint r=0;r<(count+4095)/4096;++r) {
    uint offset=r*4096+tid*4;
    float vals[4];
    for(uint j=0;j<4;++j) vals[j]=offset+j<count ? float(logits[offset+j]) : -INFINITY;
    float prev=vmax;
    for(uint j=0;j<4;++j) vmax=max(vmax,vals[j]);
    normalizer *= fast::exp(prev-vmax);
    for(uint j=0;j<4;++j) normalizer += fast::exp(vals[j]-vmax);
  }
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
