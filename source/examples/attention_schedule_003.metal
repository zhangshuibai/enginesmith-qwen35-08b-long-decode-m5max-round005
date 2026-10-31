/*
Adapted from MLX 0.32.4. Original sources and modifications are recorded in
ATTENTION_INTERFACE.md.

MIT License

Copyright © 2023 Apple Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

*/

// Session 003 isolated scheduling experiment. No arithmetic order changes.
// The parent source is prepended, providing types and constants.
template<int PHASE, int S>
[[kernel]] void attention_schedule_003(
    const device bfloat* queries [[buffer(0)]],
    const device bfloat* keys [[buffer(1)]],
    const device bfloat* values [[buffer(2)]],
    device bfloat* out [[buffer(3)]],
    device float* sums [[buffer(4)]],
    device float* maxs [[buffer(5)]],
    constant AttentionDecodeParams& p [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint gid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  constexpr int D=256, E=8, TK=4;
  typedef float U;
  const int work=tid.x*S+gid;
  const int head=work%8, block_idx=work/8;
  const int N=int(p.length);
  const int stride=blocks*D;
  keys += size_t(head/4)*p.capacity*D + block_idx*D + lane*E;
  values += size_t(head/4)*p.capacity*D + block_idx*D + lane*E;
  queries += head*D+lane*E;
  U q[E],o[E]={0};
  for (int j=0;j<E;j++) q[j]=static_cast<U>(0.0625f)*queries[j];
  U max_score=Limits<U>::finite_min,sum_exp_score=0;
  int i=block_idx;
  for (; i+(TK-1)*blocks<N;i+=TK*blocks) {
    bfloat k_reg[TK][E],v_reg[TK][E];
    for (int t=0;t<TK;t++) for (int j=0;j<E;j++) k_reg[t][j]=keys[t*stride+j];
    for (int t=0;t<TK;t++) for (int j=0;j<E;j++) v_reg[t][j]=values[t*stride+j];
    if (PHASE==0) {
      for (int t=0;t<TK;t++) {
        U score=0;
        for (int j=0;j<E;j++) score+=q[j]*static_cast<U>(k_reg[t][j]);
        score=simd_sum(score);
        U new_max=max(max_score,score);
        U factor=fast::exp(max_score-new_max);
        U exp_score=fast::exp(score-new_max);
        max_score=new_max;
        sum_exp_score=sum_exp_score*factor+exp_score;
        for (int j=0;j<E;j++) o[j]=o[j]*factor+exp_score*static_cast<U>(v_reg[t][j]);
      }
    } else {
      U scores[TK];
      for (int t=0;t<TK;t++) {
        U score=0;
        for (int j=0;j<E;j++) score+=q[j]*static_cast<U>(k_reg[t][j]);
        scores[t]=simd_sum(score);
      }
      if (PHASE==1) {
        for (int t=0;t<TK;t++) {
          U score=scores[t];
          U new_max=max(max_score,score);
          U factor=fast::exp(max_score-new_max);
          U exp_score=fast::exp(score-new_max);
          max_score=new_max;
          sum_exp_score=sum_exp_score*factor+exp_score;
          for (int j=0;j<E;j++) o[j]=o[j]*factor+exp_score*static_cast<U>(v_reg[t][j]);
        }
      } else if(PHASE==2) {
        U factors[TK],exps[TK];
        for (int t=0;t<TK;t++) {
          U new_max=max(max_score,scores[t]);
          factors[t]=fast::exp(max_score-new_max);
          exps[t]=fast::exp(scores[t]-new_max);
          max_score=new_max;
        }
        for (int t=0;t<TK;t++) {
          sum_exp_score=sum_exp_score*factors[t]+exps[t];
          for (int j=0;j<E;j++) o[j]=o[j]*factors[t]+exps[t]*static_cast<U>(v_reg[t][j]);
        }
      } else {
        for (int t=0;t<TK;t++) {
          U score=scores[t];
          if (max_score>=score) {
            U exp_score=fast::exp(score-max_score);
            sum_exp_score=sum_exp_score+exp_score;
            for (int j=0;j<E;j++) o[j]=o[j]+exp_score*static_cast<U>(v_reg[t][j]);
          } else {
            U factor=fast::exp(max_score-score);
            max_score=score;
            sum_exp_score=sum_exp_score*factor+1.0f;
            for (int j=0;j<E;j++) o[j]=o[j]*factor+static_cast<U>(v_reg[t][j]);
          }
        }
      }
    }
    keys+=TK*stride;values+=TK*stride;
  }
  for(;i<N;i+=blocks) {
    U score=0;
    for(int j=0;j<E;j++) score+=q[j]*keys[j];
    score=simd_sum(score);
    U new_max=max(max_score,score);
    U factor=fast::exp(max_score-new_max);
    U exp_score=fast::exp(score-new_max);
    max_score=new_max;
    sum_exp_score=sum_exp_score*factor+exp_score;
    for(int j=0;j<E;j++) o[j]=o[j]*factor+exp_score*values[j];
    keys+=stride;values+=stride;
  }
  const int row=head*blocks+block_idx;
  if(lane==0) {sums[row]=sum_exp_score;maxs[row]=max_score;}
  for(int j=0;j<E;j++) out[row*D+lane*E+j]=static_cast<bfloat>(o[j]);
}
#define SCHEDULE(P,S) \
template [[host_name("schedule_p" #P "_s" #S)]] [[kernel]] \
decltype(attention_schedule_003<P,S>) attention_schedule_003<P,S>;
SCHEDULE(0,1)
SCHEDULE(0,2)
SCHEDULE(0,4)
SCHEDULE(0,8)
SCHEDULE(0,16)
SCHEDULE(1,1)
SCHEDULE(1,2)
SCHEDULE(1,4)
SCHEDULE(1,8)
SCHEDULE(1,16)
SCHEDULE(2,1)
SCHEDULE(2,2)
SCHEDULE(2,4)
SCHEDULE(2,8)
SCHEDULE(2,16)
SCHEDULE(3,1)
SCHEDULE(3,2)
SCHEDULE(3,4)
SCHEDULE(3,8)
SCHEDULE(3,16)
