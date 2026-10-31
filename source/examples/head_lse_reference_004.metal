// Original greedy per-thread logsumexp walk, adapted from MLX 0.32.4.
// Copyright (c) 2023-2025 Apple Inc. See third_party/linear_mlx/LICENSE.
#include <metal_stdlib>
using namespace metal;

kernel void head_lse_partial(device const bfloat* logits [[buffer(0)]],device float* partial_max [[buffer(1)]],device float* partial_sum [[buffer(2)]],constant uint& count [[buffer(3)]],uint tid [[thread_index_in_threadgroup]]){
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
partial_max[tid]=vmax;partial_sum[tid]=normalizer;
}
