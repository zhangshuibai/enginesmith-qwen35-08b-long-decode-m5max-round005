// Tile geometry adapted from MLX 0.32.4 qmm_t_nax_tgp_impl.
// Typed dequantization is staged once per 64x64 tile. Output rows are guarded.
using namespace metal;
using namespace mlx::steel;
struct LinearShape { uint M; uint N; uint K; };

kernel void qmm_q4_nax(
    device const uint* w [[buffer(0)]],
    device const bfloat* scales [[buffer(1)]],
    device const bfloat* biases [[buffer(2)]],
    device const bfloat* x [[buffer(3)]],
    device bfloat* y [[buffer(4)]],
    constant LinearShape& p [[buffer(5)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  constexpr short BM=64, BN=64, BK=64, BKS=72;
  threadgroup bfloat weights[BN*BKS];
  uint row=gid.y*BM, col=gid.x*BN;
  short tm=32*(sg/2), tn=32*(sg%2);
  short rows=short(clamp(int(p.M)-int(row)-tm,0,32));
  short cols=short(clamp(int(p.N)-int(col)-tn,0,32));
  NAXTile<float,2,2> d;
  d.clear();
  for(uint kb=0;kb<p.K;kb+=BK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint j=tid;j<BN*BK;j+=128) {
      uint n=col+j/BK, k=kb+j%BK;
      bfloat v=0;
      if(n<p.N && k<p.K) {
        uint q=(w[n*(p.K/8)+k/8]>>(4*(k%8)))&15;
        uint g=n*(p.K/64)+k/64;
        v=bfloat(float(scales[g])*float(q)+float(biases[g]));
      }
      weights[(j/BK)*BKS+j%BK]=v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    STEEL_PRAGMA_NO_UNROLL
    for(short kk=0;kk<BK;kk+=32) {
      NAXTile<bfloat,2,2> a,b;
      device const bfloat* xp=x+(row+tm)*p.K+kb+kk;
      if(rows==32) a.load(xp,p.K);
      else a.load_safe(xp,p.K,short2(32,rows));
      b.template load<bfloat,BKS,1>(weights+tn*BKS+kk);
      tile_matmad_nax(d,a,metal::bool_constant<false>{},b,metal::bool_constant<true>{});
    }
  }
  if(rows==32 && cols==32) d.store(y+(row+tm)*p.N+col+tn,p.N);
  else d.store_safe(y+(row+tm)*p.N+col+tn,p.N,short2(cols,rows));
}
