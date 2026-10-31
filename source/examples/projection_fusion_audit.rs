#![allow(dead_code)]
#[path = "../src/gpu.rs"] mod gpu;
#[path = "../src/weights.rs"] mod weights;
use anyhow::{ensure, Result};
use gpu::{Buf, Enc, Gpu};
use objc2_metal::{MTLBuffer, MTLCommandBuffer};
use weights::{Linear, Weights};

fn next(seed:&mut u64)->u64 {
    *seed^=*seed<<13; *seed^=*seed>>7; *seed^=*seed<<17; *seed
}
fn random_bf16(n:usize,seed:&mut u64,scale:f32)->Vec<u8> {
    let mut out=Vec::with_capacity(n*2);
    for i in 0..n {
        let x=(((next(seed)>>40) as f32/(1u32<<23) as f32)-1.0)*scale;
        let bits=if i%251==0 {0u16} else if i%251==1 {0x8000} else {(x.to_bits()>>16) as u16};
        out.extend_from_slice(&bits.to_le_bytes());
    }
    out
}
fn bytes(b:&Buf)->&[u8] {
    unsafe {std::slice::from_raw_parts(b.contents().as_ptr().cast(),b.length())}
}
fn diffs(a:&Buf,b:&Buf,count:usize)->usize {
    bytes(a)[..count*2].chunks_exact(2).zip(bytes(b)[..count*2].chunks_exact(2)).filter(|(x,y)|x!=y).count()
}
fn random_linear(g:&Gpu,n:usize,k:usize,seed:&mut u64)->Result<Linear> {
    let mut codes=Vec::with_capacity(n*k/2);
    for _ in 0..n*k/8 {codes.extend_from_slice(&(next(seed) as u32).to_le_bytes());}
    Ok(Linear{w:g.from_bytes(&codes)?,s:g.from_bytes(&random_bf16(n*k/64,seed,0.05))?,
        b:g.from_bytes(&random_bf16(n*k/64,seed,0.3))?,n,k})
}
struct Projections { weights:Vec<Linear>, label:String }
fn project(g:&Gpu,e:&Enc,p:&Projections,x:&Buf,out:&[Buf],m:usize,fused:bool) {
    if fused {
        let mut bufs=Vec::new();let mut ns=Vec::new();
        for i in 0..4 {
            let valid=i<p.weights.len();let j=if valid {i}else{0};
            let w=&p.weights[j];
            bufs.extend([(&w.w,0),(&w.s,0),(&w.b,0),(&out[j],0)]);
            ns.push(if valid {w.n as u32}else{0});
        }
        bufs.push((x,0));
        g.dispatch(e,"qmv_multi_q4",&bufs,&[m as u32,ns[0],ns[1],ns[2],ns[3],p.weights[0].k as u32],
            [m,p.weights.iter().map(|w|w.n.div_ceil(8)).sum(),1],[64,1,1]);
    } else {
        for (w,y) in p.weights.iter().zip(out) {
            g.dispatch(e,"qmv_q4",&[(&w.w,0),(&w.s,0),(&w.b,0),(x,0),(y,0)],
                &[m as u32,w.n as u32,w.k as u32],[m,w.n.div_ceil(8),1],[64,1,1]);
        }
    }
}
fn run<F:FnMut(&Enc)>(g:&Gpu,reps:usize,mut f:F)->Result<f64> {
    let (cmd,enc)=g.begin()?;
    for _ in 0..reps {f(&enc);}
    let cmd=Gpu::commit(cmd,enc);Gpu::wait(&cmd)?;
    Ok((cmd.GPUEndTime()-cmd.GPUStartTime())*1e6/reps as f64)
}
fn outputs(g:&Gpu,p:&Projections,m:usize)->Result<Vec<Buf>> {
    p.weights.iter().map(|w|g.buffer(m*w.n*2)).collect()
}
fn audit(g:&Gpu,p:&Projections,m:usize,seed:&mut u64,bench:bool)->Result<()> {
    let parent=outputs(g,p,m)?;let cand=outputs(g,p,m)?;
    for scale in [0.0,0.000001,0.01,1.0,10.0,100.0] {
        let x=g.from_bytes(&random_bf16(m*p.weights[0].k,seed,scale))?;
        run(g,1,|e|project(g,e,p,&x,&parent,m,false))?;
        run(g,1,|e|project(g,e,p,&x,&cand,m,true))?;
        for (i,((a,b),w)) in parent.iter().zip(&cand).zip(&p.weights).enumerate() {
            let d=diffs(a,b,m*w.n);
            println!("{}",serde_json::json!({"check":"projection","label":p.label,"projection":i,"M":m,"N":w.n,"K":w.k,"scale":scale,"values":m*w.n,"diff":d}));
            ensure!(d==0,"projection differs:{d}");
        }
        if bench && scale==1.0 {
            for round in 0..8 {
                for fused in if round%2==0 {[false,true]}else{[true,false]} {
                    let us=run(g,128,|e|project(g,e,p,&x,if fused {&cand}else{&parent},m,fused))?;
                    println!("{}",serde_json::json!({"bench":"projection","label":p.label,"M":m,"round":round,"fused":fused,"us":us}));
                }
            }
        }
    }
    Ok(())
}
fn main()->Result<()> {
    let args:Vec<String>=std::env::args().collect();
    let bench=args.get(1).is_some_and(|s|s=="bench");
    let mut g=Gpu::new()?;
    g.compile(include_str!("../src/kernels/linear.metal"),&["qmv_q4","qmv_multi_q4"])?;
    let mut seed=0x412678b890a10f22;
    let mut all=Vec::new();
    if let Some(path)=args.get(2) {
        let mut w=Weights::load(&g,std::path::Path::new(path))?;
        for i in 0..24 {
            let mut matrices=Vec::new();
            let (kind,names)=if i%4==3 {
                ("attention",vec![("q_proj",4096),("k_proj",512),("v_proj",512)])
            } else {
                ("gdn",vec![("in_proj_qkv",6144),("in_proj_z",2048),("in_proj_a",16),("in_proj_b",16)])
            };
            for (name,n) in names {
                matrices.push(w.linear(&format!("layers.{i}.{}.{name}",if kind=="gdn" {"linear_attn"}else{"self_attn"}),n,1024)?);
            }
            all.push(Projections{weights:matrices,label:format!("{kind}-layer{i}")});
        }
    } else {
        for trial in 0..4 {
            for (ns,k) in [(vec![1,3,7,9],512),(vec![6144,2048,16,16],1024),(vec![4096,512,512],1024),(vec![9,15,17],3584)] {
                let weights=ns.iter().map(|n|random_linear(&g,*n,k,&mut seed)).collect::<Result<_>>()?;
                all.push(Projections{weights,label:format!("synthetic{trial}-{ns:?}")});
            }
        }
    }
    for p in &all {audit(&g,p,1,&mut seed,bench)?;}
    if !bench {for p in &all {audit(&g,p,3,&mut seed,false)?;}}
    if bench {
        let x=g.from_bytes(&random_bf16(3584,&mut seed,1.0))?;
        let out=all.iter().map(|p|outputs(&g,p,1)).collect::<Result<Vec<_>>>()?;
        for round in 0..8 {
            for fused in if round%2==0 {[false,true]}else{[true,false]} {
                let us=run(&g,16,|e|{for (p,o) in all.iter().zip(&out) {project(&g,e,p,&x,o,1,fused);}})?;
                println!("{}",serde_json::json!({"bench":"all-layer-stream","layers":all.len(),"round":round,"fused":fused,"us":us}));
            }
        }
    }
    Ok(())
}
