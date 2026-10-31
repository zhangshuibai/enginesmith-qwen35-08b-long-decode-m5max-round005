// qmm_t_splitk tile and reduction order from MLX 0.32.4 quantized.h.
using namespace metal;
using namespace mlx::steel;
struct SplitShape { uint M; uint N; uint K; uint splits; };

kernel void qmm_q4_splitk(
    device const uint* w [[buffer(0)]],
    device const bfloat* scales [[buffer(1)]],
    device const bfloat* biases [[buffer(2)]],
    device const bfloat* x [[buffer(3)]],
    device float* partial [[buffer(4)]],
    constant SplitShape& p [[buffer(5)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint BM=32,BN=32,BK=32,BKS=40;
  threadgroup bfloat xs[BM*BKS],ws[BN*BKS];
  BlockMMA<bfloat,float,32,32,32,2,2,false,true,40,40> mma(sg,lane);
  uint row=gid.y*BM, col=gid.x*BN;
  uint keff=p.K/p.splits;
  uint kstart=gid.z*keff;
  for(uint kb=0;kb<keff;kb+=BK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint j=tid;j<BM*BK;j+=128) {
      uint m=row+j/BK, n=col+j/BK, k=kstart+kb+j%BK;
      xs[(j/BK)*BKS+j%BK]=m<p.M ? x[m*p.K+k] : bfloat(0);
      bfloat v=0;
      if(n<p.N) {
        uint q=(w[n*(p.K/8)+k/8]>>(4*(k%8)))&15;
        uint g=n*(p.K/64)+k/64;
        v=bfloat(float(scales[g])*float(q)+float(biases[g]));
      }
      ws[(j/BK)*BKS+j%BK]=v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mma.mma(xs,ws);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  device float* out=partial+gid.z*p.M*p.N+row*p.N+col;
  short rows=min(uint(32),p.M-row),cols=min(uint(32),p.N-col);
  if(rows==32 && cols==32) mma.store_result(out,p.N);
  else mma.store_result_safe(out,p.N,short2(cols,rows));
}

kernel void qmm_q4_splitk_reduce(
    device const float* partial [[buffer(0)]],
    device bfloat* out [[buffer(1)]],
    constant SplitShape& p [[buffer(2)]],
    uint i [[thread_position_in_grid]]) {
  uint stride=p.M*p.N;
  if(i>=stride) return;
  float v=0;
  for(uint s=0;s<p.splits;++s) v+=partial[s*stride+i];
  out[i]=bfloat(v);
}
