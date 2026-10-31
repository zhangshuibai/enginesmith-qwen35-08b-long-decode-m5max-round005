"""Development-only native Metal attention parity probe; serialized GPU access."""
from pathlib import Path
import argparse,fcntl,json,subprocess
import numpy as np
import mlx.core as mx
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT.parent/'attention_probe_results';OUT.mkdir(exist_ok=True)
subprocess.run(['cargo','build','--release','--example','attention_probe','--manifest-path',str(ROOT/'Cargo.toml')],check=True)
lock=open('/tmp/engine-gpu.lock','w');fcntl.flock(lock,fcntl.LOCK_EX)
mx.random.seed(781)
def save(path,x):
    mx.eval(x);np.array(x.view(mx.uint16)).tofile(path)
def stat(d,name):
    a=np.fromfile(d/(name+'.bin'),np.uint16);b=np.fromfile(d/(name+'_ref.bin'),np.uint16)
    af=(a.astype(np.uint32)<<16).view(np.float32);bf=(b.astype(np.uint32)<<16).view(np.float32)
    return dict(different=int(np.sum(a!=b)),count=int(len(a)),max_abs=float(np.max(np.abs(af-bf))))
ap=argparse.ArgumentParser();ap.add_argument('--cases',default='1:1,1:31,1:1023,1:1024,1:4096,2:2,3:3,8:8,8:513,16:16,33:33,127:127,511:511,1023:1023,33:3000,1024:1024,2047:4095,1:131072');ap.add_argument('--output',default='results.json')
a=ap.parse_args()
results=[]
for t,n in [tuple(map(int,pair.split(':'))) for pair in a.cases.split(',')]:
    d=OUT/f'{t}_{n}';d.mkdir(exist_ok=True);c=n+32;offset=n-t
    qp=mx.random.normal((t,8,512)).astype(mx.bfloat16);kp=mx.random.normal((t,2,256)).astype(mx.bfloat16);vp=mx.random.normal((t,2,256)).astype(mx.bfloat16)
    qn=(mx.random.normal((256,))*0.1+1).astype(mx.bfloat16);kn=(mx.random.normal((256,))*0.1+1).astype(mx.bfloat16)
    kc=mx.random.normal((2,c,256)).astype(mx.bfloat16);vc=mx.random.normal((2,c,256)).astype(mx.bfloat16)
    for name,x in [('qp',qp),('kp',kp),('vp',vp),('qn',qn),('kn',kn),('kc',kc),('vc',vc)]:save(d/(name+'.bin'),x)
    q=mx.fast.rms_norm(qp[:,:,:256],qn,1e-6).transpose(1,0,2)
    q=mx.fast.rope(q,64,traditional=False,base=1e7,scale=1.,offset=offset)
    k=mx.fast.rms_norm(kp,kn,1e-6).transpose(1,0,2)
    k=mx.fast.rope(k,64,traditional=False,base=1e7,scale=1.,offset=offset)
    kc=kc.at[:,offset:n,:].add(k-kc[:,offset:n,:]) if False else mx.concatenate([kc[:,:offset,:],k,kc[:,n:,:]],axis=1)
    vc=mx.concatenate([vc[:,:offset,:],vp.transpose(1,0,2),vc[:,n:,:]],axis=1)
    o=mx.fast.scaled_dot_product_attention(q[None],kc[None,:,:n],vc[None,:,:n],scale=1/16,mask='causal' if t>1 else None)[0].transpose(1,0,2)
    gate=(o*mx.sigmoid(qp[:,:,256:])).astype(mx.bfloat16)
    for name,x in [('q',q.transpose(1,0,2)),('k',kc),('v',vc),('out',o),('gate',gate)]:save(d/(name+'_ref.bin'),x)
    r=subprocess.run([str(ROOT/'target/release/examples/attention_probe'),str(d),str(t),str(n)],check=True,capture_output=True,text=True)
    case={'shape':[t,n],'device':r.stderr.strip()};case.update({name:stat(d,name) for name in ('q','k','v','out','gate')})
    print(json.dumps(case),flush=True);results.append(case)
(OUT/a.output).write_text(json.dumps(results,indent=2))
