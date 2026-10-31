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
fn qmv(g:&Gpu,e:&Enc,w:&Linear,x:&Buf,y:&Buf,m:usize) {
    g.dispatch(e,"qmv_q4",&[(&w.w,0),(&w.s,0),(&w.b,0),(x,0),(y,0)],
        &[m as u32,w.n as u32,w.k as u32],[m,w.n.div_ceil(8),1],[64,1,1]);
}
struct MlpOut {gate:Buf,up:Buf,out:Buf}
impl MlpOut {
    fn new(g:&Gpu,m:usize,n:usize)->Result<Self> {
        Ok(Self{gate:g.buffer(m*n*2)?,up:g.buffer(m*n*2)?,out:g.buffer(m*n*2)?})
    }
}
fn mlp(g:&Gpu,e:&Enc,gate:&Linear,up:&Linear,x:&Buf,o:&MlpOut,m:usize,fused:bool) {
    if fused {
        g.dispatch(e,"qmv_gate_up_swiglu_q4",&[(&gate.w,0),(&gate.s,0),(&gate.b,0),
            (&up.w,0),(&up.s,0),(&up.b,0),(x,0),(&o.out,0)],
            &[m as u32,gate.n as u32,gate.k as u32],[m,gate.n.div_ceil(8),1],[128,1,1]);
    } else {
        qmv(g,e,gate,x,&o.gate,m); qmv(g,e,up,x,&o.up,m);
        g.dispatch(e,"swiglu_bf16",&[(&o.gate,0),(&o.up,0),(&o.out,0)],
            &[(m*gate.n) as u32],[(m*gate.n).div_ceil(256),1,1],[256,1,1]);
    }
}
fn norm(g:&Gpu,e:&Enc,a:&Buf,b:&Buf,w:&Buf,sum:&Buf,out:&Buf,m:usize,n:usize,fused:bool) {
    let p=[m as u32,n as u32,1e-6f32.to_bits()];
    if fused {
        g.dispatch(e,"add_rms_norm_bf16",&[(a,0),(b,0),(w,0),(sum,0),(out,0)],&p,
            [m,1,1],[n.div_ceil(128)*32,1,1]);
    } else {
        g.dispatch(e,"add_bf16",&[(a,0),(b,0),(sum,0)],&[(m*n) as u32],[(m*n).div_ceil(256),1,1],[256,1,1]);
        g.dispatch(e,"rms_norm_bf16",&[(sum,0),(w,0),(out,0)],&p,[m,1,1],[n.div_ceil(128)*32,1,1]);
    }
}
fn run<F:FnMut(&Enc)>(g:&Gpu,reps:usize,mut f:F)->Result<f64> {
    let (cmd,enc)=g.begin()?;
    for _ in 0..reps {f(&enc);}
    let cmd=Gpu::commit(cmd,enc); Gpu::wait(&cmd)?;
    Ok((cmd.GPUEndTime()-cmd.GPUStartTime())*1e6/reps as f64)
}
fn audit_mlp(g:&Gpu,gate:&Linear,up:&Linear,m:usize,seed:&mut u64,label:&str,bench:bool)->Result<()> {
    let parent=MlpOut::new(g,m,gate.n)?;let cand=MlpOut::new(g,m,gate.n)?;
    for scale in [0.0,0.001,0.1,1.0,10.0,100.0] {
        let x=g.from_bytes(&random_bf16(m*gate.k,seed,scale))?;
        run(g,1,|e|mlp(g,e,gate,up,&x,&parent,m,false))?;
        run(g,1,|e|mlp(g,e,gate,up,&x,&cand,m,true))?;
        let d=diffs(&parent.out,&cand.out,m*gate.n);
        println!("{}",serde_json::json!({"check":"mlp","weights":label,"M":m,"N":gate.n,"K":gate.k,"scale":scale,"values":m*gate.n,"diff":d}));
        ensure!(d==0,"MLP differs: {d}");
        if bench && scale==1.0 {
            for round in 0..8 {
                for fused in if round%2==0 {[false,true]} else {[true,false]} {
                    let o=if fused {&cand}else{&parent};
                    let us=run(g,128,|e|mlp(g,e,gate,up,&x,o,m,fused))?;
                    println!("{}",serde_json::json!({"bench":"mlp","weights":label,"M":m,"N":gate.n,"K":gate.k,"round":round,"fused":fused,"us":us}));
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
    g.compile(include_str!("../src/kernels/linear.metal"),
        &["qmv_q4","swiglu_bf16","qmv_gate_up_swiglu_q4","add_bf16","rms_norm_bf16","add_rms_norm_bf16"])?;
    let mut seed=0x4a78b901df68e123;
    for trial in 0..8 {
        for (m,n) in [(1usize,3usize),(1,128),(1,256),(1,1024),(1,1025),(1,4096),(3,1024),(17,1024)] {
            if bench && (m,n)!=(1,1024) {continue;}
            for scale in [0.0,1e-10,0.01,1.0,10.0,100.0] {
                let ab=random_bf16(m*n,&mut seed,scale);let bb=random_bf16(m*n,&mut seed,scale);
                let a=g.from_bytes(&ab)?;let b=g.from_bytes(&bb)?;
                let gain=g.from_bytes(&random_bf16(n,&mut seed,1.5))?;
                let s0=g.buffer(m*n*2)?;let s1=g.buffer(m*n*2)?;
                let y0=g.buffer(m*n*2)?;let y1=g.buffer(m*n*2)?;
                run(&g,1,|e|norm(&g,e,&a,&b,&gain,&s0,&y0,m,n,false))?;
                run(&g,1,|e|norm(&g,e,&a,&b,&gain,&s1,&y1,m,n,true))?;
                let sd=diffs(&s0,&s1,m*n);let yd=diffs(&y0,&y1,m*n);
                println!("{}",serde_json::json!({"check":"norm","trial":trial,"M":m,"N":n,"scale":scale,"sum_diff":sd,"norm_diff":yd,"values":m*n}));
                ensure!(sd+yd==0,"RMS differs");
                for alias in 0..2 {
                    let aa=g.from_bytes(&ab)?;let bb=g.from_bytes(&bb)?;
                    run(&g,1,|e|norm(&g,e,&aa,&bb,&gain,if alias==0 {&aa}else{&bb},&y1,m,n,true))?;
                    ensure!(diffs(&s0,if alias==0 {&aa}else{&bb},m*n)+diffs(&y0,&y1,m*n)==0,"alias RMS differs");
                }
                if bench && trial==0 && scale==1.0 {
                    for round in 0..8 {
                        for fused in if round%2==0 {[false,true]} else {[true,false]} {
                            let us=run(&g,512,|e|norm(&g,e,&a,&b,&gain,&s1,&y1,m,n,fused))?;
                            println!("{}",serde_json::json!({"bench":"norm","M":m,"N":n,"round":round,"fused":fused,"us":us}));
                        }
                    }
                }
            }
        }
    }
    if let Some(path)=args.get(2) {
        let mut w=Weights::load(&g,std::path::Path::new(path))?;
        for i in 0..24 {
            let gate=w.linear(&format!("layers.{i}.mlp.gate_proj"),3584,1024)?;
            let up=w.linear(&format!("layers.{i}.mlp.up_proj"),3584,1024)?;
            audit_mlp(&g,&gate,&up,1,&mut seed,&format!("layer{i}"),bench && i==0)?;
        }
    } else {
        for trial in 0..4 {
            for (m,n,k) in [(1,1,512),(1,7,1024),(1,8,1024),(1,9,1024),(1,16,2048),(1,3584,1024),(2,3584,1024),(3,32,3584)] {
                if bench && (m,n,k)!=(1,3584,1024) {continue;}
                let gate=random_linear(&g,n,k,&mut seed)?;let up=random_linear(&g,n,k,&mut seed)?;
                audit_mlp(&g,&gate,&up,m,&mut seed,&format!("synthetic{trial}"),bench)?;
            }
        }
    }
    Ok(())
}
