#![allow(dead_code)]
#[path = "../src/gpu.rs"] mod gpu;
#[path = "../src/weights.rs"] mod weights;
use anyhow::{ensure, Result};
use gpu::{Buf, Enc, Gpu};
use objc2_metal::{MTLBuffer, MTLCommandBuffer};
use weights::{Linear, Weights};
fn next(s:&mut u64)->u64 {*s^=*s<<13;*s^=*s>>7;*s^=*s<<17;*s}
fn randbf(n:usize,s:&mut u64,scale:f32)->Vec<u8>{let mut v=Vec::with_capacity(n*2);for i in 0..n{let f=(((next(s)>>40)as f32/(1u32<<23)as f32)-1.0)*scale;let b=match i%251{0=>0u16,1=>0x8000,_=>(f.to_bits()>>16)as u16};v.extend_from_slice(&b.to_le_bytes());}v}
fn bytes(b:&Buf)->&[u8]{unsafe{std::slice::from_raw_parts(b.contents().as_ptr().cast(),b.length())}}
fn diffs(a:&Buf,b:&Buf,n:usize)->usize{bytes(a)[..n*2].chunks_exact(2).zip(bytes(b)[..n*2].chunks_exact(2)).filter(|(a,b)|a!=b).count()}
fn run<F:FnMut(&Enc)>(g:&Gpu,reps:usize,mut f:F)->Result<f64>{let(c,e)=g.begin()?;for _ in 0..reps{f(&e);}let c=Gpu::commit(c,e);Gpu::wait(&c)?;Ok((c.GPUEndTime()-c.GPUStartTime())*1e6/reps as f64)}
struct Outputs{norm:Buf,out:Buf}
fn dispatch(g:&Gpu,e:&Enc,gate:&Linear,up:&Linear,gain:&Buf,x:&Buf,o:&Outputs,groups:usize,fused:bool){
 if !fused{g.dispatch(e,"rms_norm_bf16",&[(x,0),(gain,0),(&o.norm,0)],&[1,1024,1e-6f32.to_bits()],[1,1,1],[256,1,1]);}
 let actual=if groups==0 {4}else{groups};let name=if groups==0 {"qmv_gate_up_swiglu_q4".to_owned()}else{format!("qmv_mlp_g{}{}",groups,if fused {"_norm"}else{""})};
 let mut bufs=vec![(&gate.w,0),(&gate.s,0),(&gate.b,0),(&up.w,0),(&up.s,0),(&up.b,0),(if fused{x}else{&o.norm},0),(&o.out,0)];if fused{bufs.push((gain,0));bufs.push((&o.norm,0));}
 g.dispatch(e,&name,&bufs,&[1,gate.n as u32,1024],[1,gate.n.div_ceil(actual*2),1],[actual*32,1,1]);
}
fn main()->Result<()>{let args:Vec<String>=std::env::args().collect();let bench=args[1]=="bench";let mut g=Gpu::new()?;g.compile(include_str!("../src/kernels/linear.metal"),&["rms_norm_bf16","qmv_gate_up_swiglu_q4"])?;
 g.compile(include_str!("../src/kernels/linear_norm.metal"),&["qmv_mlp_g8_norm"])?;
 let mut w=Weights::load(&g,std::path::Path::new(&args[2]))?;let mut seed=0x17231afed34;let a=Outputs{norm:g.buffer(2048)?,out:g.buffer(7168)?};let b=Outputs{norm:g.buffer(2048)?,out:g.buffer(7168)?};
 for i in 0..24 {let gate=w.linear(&format!("layers.{i}.mlp.gate_proj"),3584,1024)?;let up=w.linear(&format!("layers.{i}.mlp.up_proj"),3584,1024)?;let gain=w.take(&format!("layers.{i}.post_attention_layernorm.weight"),&[1024],"BF16")?;
 for scale in if bench{vec![1.0]}else{vec![0.0,0.001,0.1,1.0,10.0,100.0]} {let x=g.from_bytes(&randbf(1024,&mut seed,scale))?;run(&g,1,|e|dispatch(&g,e,&gate,&up,&gain,&x,&a,0,false))?;
 for groups in [8] {for fused in [true] {if groups==4&&fused{continue;}run(&g,1,|e|dispatch(&g,e,&gate,&up,&gain,&x,&b,groups,fused))?;let d=diffs(&a.out,&b.out,3584);let nd=diffs(&a.norm,&b.norm,1024);println!("{}",serde_json::json!({"check":"norm_mlp","layer":i,"scale":scale,"groups":groups,"fused":fused,"diff":d,"norm_diff":nd}));ensure!(d+nd==0,"norm mlp differs");}}
 if bench&&i==0{for round in 0..12{let mut vars=vec![(0,false),(8,true)];if round%2==1{vars.reverse();}for(groups,fused)in vars{let us=run(&g,128,|e|dispatch(&g,e,&gate,&up,&gain,&x,&b,groups,fused))?;println!("{}",serde_json::json!({"bench":"norm_mlp","groups":groups,"fused":fused,"round":round,"us":us}));}}}
 }
 }Ok(())}
