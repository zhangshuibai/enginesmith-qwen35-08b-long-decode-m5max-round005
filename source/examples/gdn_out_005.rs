//! Primitive audit and isolated timing; caller holds the shared GPU lock.
#![allow(dead_code)]
#[path="../src/gpu.rs"] mod gpu;
#[path="../src/weights.rs"] mod weights;
use anyhow::{ensure,Result};
use gpu::{Buf,Enc,Gpu};
use objc2_metal::{MTLBuffer,MTLCommandBuffer};
use weights::{Linear,Weights};
fn next(s:&mut u64)->u64 {*s^=*s<<13;*s^=*s>>7;*s^=*s<<17;*s}
fn randbf(n:usize,s:&mut u64,scale:f32)->Vec<u8>{let mut v=Vec::with_capacity(n*2);for i in 0..n{let f=(((next(s)>>40)as f32/(1u32<<23)as f32)-1.0)*scale;let b=match i%251{0=>0u16,1=>0x8000,_=>(f.to_bits()>>16)as u16};v.extend_from_slice(&b.to_le_bytes());}v}
fn bytes(b:&Buf)->&[u8]{unsafe{std::slice::from_raw_parts(b.contents().as_ptr().cast(),b.length())}}
fn diffs(a:&Buf,b:&Buf,n:usize)->usize{bytes(a)[..n*2].chunks_exact(2).zip(bytes(b)[..n*2].chunks_exact(2)).filter(|(a,b)|a!=b).count()}
fn run<F:FnMut(&Enc)>(g:&Gpu,reps:usize,mut f:F)->Result<f64>{let(c,e)=g.begin()?;for _ in 0..reps{f(&e);}let c=Gpu::commit(c,e);Gpu::wait(&c)?;Ok((c.GPUEndTime()-c.GPUStartTime())*1e6/reps as f64)}
struct Outputs{norm:Buf,out:Buf}
struct Layer{out:Linear,gain:Buf,y:Buf,z:Buf,residual:Buf}
fn dispatch(g:&Gpu,e:&Enc,l:&Layer,o:&Outputs,groups:usize){
 if groups==0 {
  g.dispatch(e,"gdn_norm_gate",&[(&l.y,0),(&l.z,0),(&l.gain,0),(&o.norm,0)],&[1],[16,1,1],[32,1,1]);
  g.dispatch(e,"qmv_add_q4",&[(&l.out.w,0),(&l.out.s,0),(&l.out.b,0),(&o.norm,0),(&l.residual,0),(&o.out,0)],&[1,1024,2048],[1,128,1],[64,1,1]);
 } else {
  g.dispatch(e,&format!("gdn_out_g{groups}_005"),&[(&l.out.w,0),(&l.out.s,0),(&l.out.b,0),(&l.y,0),(&l.z,0),(&l.gain,0),(&l.residual,0),(&o.out,0),(&o.norm,0)],&[1,1024,2048],[1,1024/(groups*4),1],[groups*32,1,1]);
 }
}
fn main()->Result<()> {
 let args:Vec<String>=std::env::args().collect();let bench=args[1]=="bench";
 let mut g=Gpu::new()?;
 g.compile(include_str!("../src/kernels/linear.metal"),&["qmv_add_q4"])?;
 g.compile(include_str!("../src/kernels/gdn.metal"),&["gdn_norm_gate"])?;
 g.compile(include_str!("gdn_out_005.metal"),&["gdn_out_g2_005","gdn_out_g4_005","gdn_out_g8_005","gdn_out_g16_005"])?;
 let mut w=Weights::load(&g,std::path::Path::new(&args[2]))?;
 let mut seed=0x9187231afed34;
 let a=Outputs{norm:g.buffer(4096)?,out:g.buffer(2048)?};
 let b=Outputs{norm:g.buffer(4096)?,out:g.buffer(2048)?};
 let mut layers=Vec::new();
 for i in 0..24 {if i%4==3 {continue;}
  let out=w.linear(&format!("layers.{i}.linear_attn.out_proj"),1024,2048)?;
  let gain=w.take(&format!("layers.{i}.linear_attn.norm.weight"),&[128],"BF16")?;
  let mut layer=Layer{out,gain,y:g.buffer(4096)?,z:g.buffer(4096)?,residual:g.buffer(2048)?};
  for scale in [0.0,0.001,0.1,1.0,10.0,100.0] {
   layer.y=g.from_bytes(&randbf(2048,&mut seed,scale))?;
   layer.z=g.from_bytes(&randbf(2048,&mut seed,scale))?;
   layer.residual=g.from_bytes(&randbf(1024,&mut seed,scale))?;
   run(&g,1,|e|dispatch(&g,e,&layer,&a,0))?;
   for groups in [2,4,8,16] {
    run(&g,1,|e|dispatch(&g,e,&layer,&b,groups))?;
    let d=diffs(&a.out,&b.out,1024);let nd=diffs(&a.norm,&b.norm,2048);
    println!("{}",serde_json::json!({"check":"gdn_out","layer":i,"scale":scale,"groups":groups,"out_diff":d,"norm_diff":nd}));
    ensure!(d+nd==0,"gdn norm+output differs");
   }
  }
  layers.push(layer);
 }
 if bench {
  for groups in [0,2,4,8,16] {run(&g,32,|e|dispatch(&g,e,&layers[0],&b,groups))?;}
  for stream in [false,true] {for round in 0..24 {
   let mut variants=vec![0,2,4,8,16];if round%2==1 {variants.reverse();}
   for groups in variants {
    let us=run(&g,if stream {32}else{256},|e| {if stream {for l in &layers {dispatch(&g,e,l,&b,groups);}}else{dispatch(&g,e,&layers[0],&b,groups);}})?;
    println!("{}",serde_json::json!({"bench":"gdn_out","stream":stream,"groups":groups,"round":round,"us":us}));
   }
  }}
 }
 Ok(())
}
