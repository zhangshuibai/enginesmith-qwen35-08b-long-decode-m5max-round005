"""Development-only Metal/reference comparison. Requires installed MLX + numpy."""
from pathlib import Path
import fcntl
import json
import subprocess
import sys
import numpy as np
import mlx.core as mx
import mlx.nn as nn

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT.parent/'linear_probe_results'
OUT.mkdir(exist_ok=True)
lock=open('/tmp/engine-gpu.lock','w')
fcntl.flock(lock,fcntl.LOCK_EX)
mx.random.seed(123)

@mx.compile
def swiglu(g,u): return nn.silu(g)*u

def save(p,a):
    mx.eval(a)
    a=np.array(a.view(mx.uint16) if a.dtype==mx.bfloat16 else a)
    a.tofile(p)

results=[]
for m,n,k in [(1,64,1024),(5,16,1024),(33,16,1024),(25,3584,1024),(13,6144,1024),(63,512,1024),(128,1024,3584),(2048,16,1024),(2048,3584,1024)]:
    d=OUT/f'{m}_{n}_{k}';d.mkdir(exist_ok=True)
    w,s,b=mx.quantize(mx.random.normal((n,k))*0.05,group_size=64,bits=4)
    s=s.astype(mx.bfloat16);b=b.astype(mx.bfloat16)
    x=mx.random.normal((m,k)).astype(mx.bfloat16)
    gain=(mx.random.normal((k,))*0.1+1).astype(mx.bfloat16)
    gate=(mx.random.normal((m,k))*2).astype(mx.bfloat16)
    up=mx.random.normal((m,k)).astype(mx.bfloat16)
    ref=mx.quantized_matmul(x,w,s,b,transpose=True,group_size=64,bits=4)
    refs={'qmv_q4':ref,'qmm_q4_nax':ref,'qmm_q4_splitk':ref,'rms':mx.fast.rms_norm(x,gain,1e-6),'swiglu':swiglu(gate,up)}
    refs['embedding']=mx.dequantize(w,s,b,group_size=64,bits=4)[mx.arange(m)%n]
    logits=(mx.random.normal((248320,))*3+4).astype(mx.bfloat16)
    save(d/'logits.bin',logits)
    next_token=mx.argmax(logits-mx.logsumexp(logits)).item()
    for name,a in [('x',x),('w',w),('s',s),('b',b),('gain',gain),('gate',gate),('up',up)]: save(d/(name+'.bin'),a)
    for name,a in refs.items():save(d/(name+'_ref.bin'),a)
    subprocess.run([str(ROOT/'target/release/linear_probe'),str(d),str(m),str(n),str(k)],check=True)
    case={'shape':[m,n,k]}
    case['greedy']={'got':int(np.fromfile(d/'token.bin',np.uint32)[0]),'want':next_token}
    for name in refs:
        got=np.fromfile(d/(name+'.bin'),np.uint16);want=np.fromfile(d/(name+'_ref.bin'),np.uint16)
        gf=(got.astype(np.uint32)<<16).view(np.float32);wf=(want.astype(np.uint32)<<16).view(np.float32)
        case[name]={'different':int(np.sum(got!=want)),'count':int(got.size),'max_abs':float(np.max(np.abs(gf-wf)))}
    print(json.dumps(case),flush=True);results.append(case)
(OUT/'results.json').write_text(json.dumps(results,indent=2))
