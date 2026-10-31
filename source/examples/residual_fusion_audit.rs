#![allow(dead_code)]
#[path = "../src/gpu.rs"] mod gpu;
#[path = "../src/weights.rs"] mod weights;
use anyhow::{ensure,Result};
use gpu::{Buf,Enc,Gpu};
use objc2_metal::{MTLBuffer,MTLCommandBuffer};
use weights::{Linear,Weights};

fn next(seed:&mut u64)->u64 {
    *seed^=*seed<<13;*seed^=*seed>>7;*seed^=*seed<<17;*seed
}
fn random_bf16(n:usize,seed:&mut u64,scale:f32)->Vec<u8> {
    let mut out=Vec::with_capacity(n*2);
    for i in 0..n {
        let x=(((next(seed)>>40) as f32/(1u32<<23) as f32)-1.0)*scale;
        let bits=if i%251==0 {0u16}else if i%251==1 {0x8000}else{(x.to_bits()>>16) as u16};
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
struct Outputs {projected:Buf,sum:Buf,norm:Buf}
impl Outputs {
    fn new(g:&Gpu,count:usize)->Result<Self> {
        Ok(Self{projected:g.buffer(count*2)?,sum:g.buffer(count*2)?,norm:g.buffer(count*2)?})
    }
}
fn chain(g:&Gpu,e:&Enc,w:&Linear,x:&Buf,h:&Buf,gain:&Buf,o:&Outputs,m:usize,variant:usize,alias:bool) {
    let lp=[m as u32,w.n as u32,w.k as u32];let np=[m as u32,w.n as u32,1e-6f32.to_bits()];
    let sum=if alias {h}else{&o.sum};
    if variant==2 {
        g.dispatch(e,"qmv_add_q4",&[(&w.w,0),(&w.s,0),(&w.b,0),(x,0),(h,0),(sum,0)],
            &lp,[m,w.n.div_ceil(8),1],[64,1,1]);
    } else {
        g.dispatch(e,"qmv_q4",&[(&w.w,0),(&w.s,0),(&w.b,0),(x,0),(&o.projected,0)],
            &lp,[m,w.n.div_ceil(8),1],[64,1,1]);
        if variant==1 {
            g.dispatch(e,"add_rms_norm_bf16",&[(h,0),(&o.projected,0),(gain,0),(sum,0),(&o.norm,0)],
                &np,[m,1,1],[w.n.div_ceil(128)*32,1,1]);
        } else {
            g.dispatch(e,"add_bf16",&[(h,0),(&o.projected,0),(sum,0)],&[(m*w.n) as u32],
                [(m*w.n).div_ceil(256),1,1],[256,1,1]);
        }
    }
    if variant!=1 {
        g.dispatch(e,"rms_norm_bf16",&[(sum,0),(gain,0),(&o.norm,0)],&np,
            [m,1,1],[w.n.div_ceil(128)*32,1,1]);
    }
}
fn run<F:FnMut(&Enc)>(g:&Gpu,reps:usize,mut f:F)->Result<f64> {
    let (cmd,enc)=g.begin()?;for _ in 0..reps {f(&enc);}
    let cmd=Gpu::commit(cmd,enc);Gpu::wait(&cmd)?;
    Ok((cmd.GPUEndTime()-cmd.GPUStartTime())*1e6/reps as f64)
}
fn audit(g:&Gpu,w:&Linear,label:&str,m:usize,seed:&mut u64,bench:bool)->Result<()> {
    let parent=Outputs::new(g,m*w.n)?;let cand=Outputs::new(g,m*w.n)?;
    for scale in [0.0,0.000001,0.01,1.0,10.0,100.0] {
        let x=g.from_bytes(&random_bf16(m*w.k,seed,scale))?;
        let hb=random_bf16(m*w.n,seed,scale);let h=g.from_bytes(&hb)?;
        let gain=g.from_bytes(&random_bf16(w.n,seed,1.5))?;
        run(g,1,|e|chain(g,e,w,&x,&h,&gain,&parent,m,0,false))?;
        for variant in 1..3 {
            run(g,1,|e|chain(g,e,w,&x,&h,&gain,&cand,m,variant,false))?;
            let sd=diffs(&parent.sum,&cand.sum,m*w.n);let nd=diffs(&parent.norm,&cand.norm,m*w.n);
            println!("{}",serde_json::json!({"check":"residual","label":label,"M":m,"N":w.n,"K":w.k,"variant":variant,"scale":scale,"values":m*w.n,"sum_diff":sd,"norm_diff":nd}));
            ensure!(sd+nd==0,"residual differs");
            let ha=g.from_bytes(&hb)?;
            run(g,1,|e|chain(g,e,w,&x,&ha,&gain,&cand,m,variant,true))?;
            ensure!(diffs(&parent.sum,&ha,m*w.n)+diffs(&parent.norm,&cand.norm,m*w.n)==0,"residual alias differs");
        }
        if bench && scale==1.0 {
            for round in 0..8 {
                for variant in if round%2==0 {[0,1,2]}else{[2,1,0]} {
                    let us=run(g,128,|e|chain(g,e,w,&x,&h,&gain,&cand,m,variant,false))?;
                    println!("{}",serde_json::json!({"bench":"residual","label":label,"M":m,"N":w.n,"K":w.k,"round":round,"variant":variant,"us":us}));
                }
            }
        }
    }
    Ok(())
}
fn main()->Result<()> {
    let args:Vec<String>=std::env::args().collect();let bench=args.get(1).is_some_and(|x|x=="bench");
    let mut g=Gpu::new()?;
    g.compile(include_str!("../src/kernels/linear.metal"),&["qmv_q4","qmv_add_q4","add_bf16","rms_norm_bf16","add_rms_norm_bf16"])?;
    let mut seed=0x891245abd1716119;let mut all=Vec::new();
    if let Some(path)=args.get(2) {
        let mut w=Weights::load(&g,std::path::Path::new(path))?;
        for i in 0..24 {
            let mixer=if i%4==3 {"self_attn.o_proj"}else{"linear_attn.out_proj"};
            all.push((format!("mixer{i}"),w.linear(&format!("layers.{i}.{mixer}"),1024,2048)?));
            all.push((format!("down{i}"),w.linear(&format!("layers.{i}.mlp.down_proj"),1024,3584)?));
        }
    } else {
        for trial in 0..4 {for (n,k) in [(1,512),(7,512),(9,1024),(128,2048),(1024,2048),(1024,3584)] {
            all.push((format!("synthetic{trial}-{n}-{k}"),random_linear(&g,n,k,&mut seed)?));
        }}
    }
    for (label,w) in &all {audit(&g,w,label,1,&mut seed,bench)?;}
    if !bench {for (label,w) in &all {audit(&g,w,label,3,&mut seed,false)?;}}
    if bench {
        let x=g.from_bytes(&random_bf16(3584,&mut seed,1.0))?;
        let h=g.from_bytes(&random_bf16(1024,&mut seed,1.0))?;
        let gain=g.from_bytes(&random_bf16(1024,&mut seed,1.5))?;
        let o=Outputs::new(&g,1024)?;
        for round in 0..8 {for variant in if round%2==0 {[0,1,2]}else{[2,1,0]} {
            let us=run(&g,16,|e|{for (_,w) in &all {chain(&g,e,w,&x,&h,&gain,&o,1,variant,false);}})?;
            println!("{}",serde_json::json!({"bench":"all-layer-stream","layers":all.len(),"round":round,"variant":variant,"us":us}));
        }}
    }
    Ok(())
}
