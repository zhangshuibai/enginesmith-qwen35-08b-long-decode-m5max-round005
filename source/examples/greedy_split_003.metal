// Parallel decomposition of the baseline MLX-derived BF16 log-softmax argmax.
// Each SIMD group retains its original 1024-thread-group logical lane indices,
// element walk, FP32 reductions, and BF16 logsumexp/subtraction boundaries.
// Copyright (c) 2023-2025 Apple Inc. See third_party/linear_mlx/LICENSE.
#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

kernel void greedy_partial(
    device const bfloat* logits [[buffer(0)]],
    device float* partial [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  uint tid=group*32+lane;
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
  if(lane==0) {partial[group]=vmax; partial[32+group]=normalizer;}
}

kernel void greedy_candidates(
    device const bfloat* logits [[buffer(0)]],
    device const float* partial [[buffer(1)]],
    device float* bests [[buffer(2)]],
    device uint* ids [[buffer(3)]],
    constant uint& count [[buffer(4)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  float prev=partial[lane];
  float vmax=simd_max(prev);
  float normalizer=partial[32+lane]*fast::exp(prev-vmax);
  normalizer=simd_sum(normalizer);
  bfloat lse=bfloat(log(normalizer)+vmax);
  uint tid=group*32+lane;
  float best=-INFINITY;
  uint id=0xffffffff;
  for(uint i=tid;i<count;i+=1024) {
    float v=float(bfloat(float(logits[i])-float(lse)));
    if(v>best || (v==best && i<id)) {best=v; id=i;}
  }
  best=simd_max(best);
  id=simd_min(float(bfloat(float(logits[min(id,count-1)])-float(lse)))==best ? id : 0xffffffff);
  if(lane==0) {bests[group]=best; ids[group]=id;}
}

kernel void greedy_finish(
    device const float* bests [[buffer(0)]],
    device const uint* ids [[buffer(1)]],
    device uint* token [[buffer(2)]],
    uint lane [[thread_index_in_simdgroup]]) {
  float best=simd_max(bests[lane]);
  uint id=simd_min(bests[lane]==best ? ids[lane] : 0xffffffff);
  if(lane==0) token[0]=id;
}
