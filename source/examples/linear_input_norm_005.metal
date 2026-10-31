// Experimental exact input RMS + grouped projections. Parent QMV/normalization
// adapted from MLX 0.32.4, Apple Inc.; see third_party/linear_mlx/LICENSE.
#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;
struct LinearShape { uint M; uint N; uint K; };
struct MultiLinearShape { uint M; uint N0; uint N1; uint N2; uint N3; uint K; };

kernel void qmv_multi_g8(
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
  if (group >= (n+31)/32) {
    group -= (n+31)/32;
    n=shape.N1;w=w1;scales=s1;biases=b1;y=y1;
    if (group >= (n+31)/32) {
      group -= (n+31)/32;
      n=shape.N2;w=w2;scales=s2;biases=b2;y=y2;
      if (group >= (n+31)/32) {
        group -= (n+31)/32;
        n=shape.N3;w=w3;scales=s3;biases=b3;y=y3;
      }
    }
  }
  const LinearShape p={shape.M,n,shape.K};
  uint row0 = group * 32 + sg * 4;
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


kernel void qmv_multi_g8_norm(
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
    constant MultiLinearShape& shape [[buffer(19)]],
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
  uint group=gid.y;
  uint n=shape.N0;
  device const uint* w=w0;
  device const bfloat* scales=s0;
  device const bfloat* biases=b0;
  device bfloat* y=y0;
  if (group >= (n+31)/32) {
    group -= (n+31)/32;
    n=shape.N1;w=w1;scales=s1;biases=b1;y=y1;
    if (group >= (n+31)/32) {
      group -= (n+31)/32;
      n=shape.N2;w=w2;scales=s2;biases=b2;y=y2;
      if (group >= (n+31)/32) {
        group -= (n+31)/32;
        n=shape.N3;w=w3;scales=s3;biases=b3;y=y3;
      }
    }
  }
  const LinearShape p={shape.M,n,shape.K};
  uint row0 = group * 32 + sg * 4;
  if (row0 >= p.N || gid.x >= p.M) return;
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
    if (lane == 0 && row0+r < p.N) y[gid.x*p.N+row0+r] = bfloat(value);
  }
}


kernel void qmv_multi_g16(
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
  if (group >= (n+63)/64) {
    group -= (n+63)/64;
    n=shape.N1;w=w1;scales=s1;biases=b1;y=y1;
    if (group >= (n+63)/64) {
      group -= (n+63)/64;
      n=shape.N2;w=w2;scales=s2;biases=b2;y=y2;
      if (group >= (n+63)/64) {
        group -= (n+63)/64;
        n=shape.N3;w=w3;scales=s3;biases=b3;y=y3;
      }
    }
  }
  const LinearShape p={shape.M,n,shape.K};
  uint row0 = group * 64 + sg * 4;
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


kernel void qmv_multi_g16_norm(
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
    constant MultiLinearShape& shape [[buffer(19)]],
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
  uint group=gid.y;
  uint n=shape.N0;
  device const uint* w=w0;
  device const bfloat* scales=s0;
  device const bfloat* biases=b0;
  device bfloat* y=y0;
  if (group >= (n+63)/64) {
    group -= (n+63)/64;
    n=shape.N1;w=w1;scales=s1;biases=b1;y=y1;
    if (group >= (n+63)/64) {
      group -= (n+63)/64;
      n=shape.N2;w=w2;scales=s2;biases=b2;y=y2;
      if (group >= (n+63)/64) {
        group -= (n+63)/64;
        n=shape.N3;w=w3;scales=s3;biases=b3;y=y3;
      }
    }
  }
  const LinearShape p={shape.M,n,shape.K};
  uint row0 = group * 64 + sg * 4;
  if (row0 >= p.N || gid.x >= p.M) return;
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
    if (lane == 0 && row0+r < p.N) y[gid.x*p.N+row0+r] = bfloat(value);
  }
}


kernel void qmv_multi_g32(
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
  if (group >= (n+127)/128) {
    group -= (n+127)/128;
    n=shape.N1;w=w1;scales=s1;biases=b1;y=y1;
    if (group >= (n+127)/128) {
      group -= (n+127)/128;
      n=shape.N2;w=w2;scales=s2;biases=b2;y=y2;
      if (group >= (n+127)/128) {
        group -= (n+127)/128;
        n=shape.N3;w=w3;scales=s3;biases=b3;y=y3;
      }
    }
  }
  const LinearShape p={shape.M,n,shape.K};
  uint row0 = group * 128 + sg * 4;
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


kernel void qmv_multi_g32_norm(
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
    constant MultiLinearShape& shape [[buffer(19)]],
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
  uint group=gid.y;
  uint n=shape.N0;
  device const uint* w=w0;
  device const bfloat* scales=s0;
  device const bfloat* biases=b0;
  device bfloat* y=y0;
  if (group >= (n+127)/128) {
    group -= (n+127)/128;
    n=shape.N1;w=w1;scales=s1;biases=b1;y=y1;
    if (group >= (n+127)/128) {
      group -= (n+127)/128;
      n=shape.N2;w=w2;scales=s2;biases=b2;y=y2;
      if (group >= (n+127)/128) {
        group -= (n+127)/128;
        n=shape.N3;w=w3;scales=s3;biases=b3;y=y3;
      }
    }
  }
  const LinearShape p={shape.M,n,shape.K};
  uint row0 = group * 128 + sg * 4;
  if (row0 >= p.N || gid.x >= p.M) return;
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
    if (lane == 0 && row0+r < p.N) y[gid.x*p.N+row0+r] = bfloat(value);
  }
}

