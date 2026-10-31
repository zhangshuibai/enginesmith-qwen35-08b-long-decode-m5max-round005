//! Isolated parity and timing of input-projection/convolution fusion.
//! Caller holds the shared GPU lock; no production route is changed here.
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
fn copy(dst:&Buf,src:&Buf){assert_eq!(dst.length(),src.length());unsafe{std::ptr::copy_nonoverlapping(src.contents().as_ptr(),dst.contents().as_ptr(),src.length());}}
fn run<F:FnMut(&Enc)>(g:&Gpu,reps:usize,mut f:F)->Result<f64>{let(c,e)=g.begin()?;for _ in 0..reps{f(&e);}let c=Gpu::commit(c,e);Gpu::wait(&c)?;Ok((c.GPUEndTime()-c.GPUStartTime())*1e6/reps as f64)}
struct Layer{ws:Vec<Linear>,gain:Buf,conv:Buf,alog:Buf,dt:Buf}
struct Outputs{norm:Buf,y:Vec<Buf>,history:Buf,conv:Buf,g:Buf,beta:Buf}
impl Outputs{
 fn new(g:&Gpu,seed:&mut u64)->Result<Self>{let mut h=randbf(18432,seed,1.0);h.extend(vec![0xa5;64]);let o=Self{norm:g.buffer(2048+64)?,y:[6144,2048,16,16].iter().map(|n|g.buffer(n*2+64)).collect::<Result<_>>()?,history:g.from_bytes(&h)?,conv:g.buffer(12288+64)?,g:g.buffer(64+64)?,beta:g.buffer(32+64)?};for(_,b,n,_)in o.all(){if n!=36864{unsafe{std::ptr::write_bytes(b.contents().as_ptr().cast::<u8>(),0xa5,b.length());}}}Ok(o)}
 fn all(&self)->Vec<(&str,&Buf,usize,usize)>{vec![("norm",&self.norm,2048,2),("qkv",&self.y[0],12288,2),("z",&self.y[1],4096,2),("a",&self.y[2],32,2),("b",&self.y[3],32,2),("history",&self.history,36864,2),("conv",&self.conv,12288,2),("g",&self.g,64,4),("beta",&self.beta,32,2)]}
}
fn dispatch(g:&Gpu,e:&Enc,l:&Layer,x:&Buf,o:&Outputs,groups:usize){
 let mut bufs=vec![];for i in 0..4{let w=&l.ws[i];bufs.extend([(&w.w,0),(&w.s,0),(&w.b,0),(&o.y[i],0)]);}bufs.extend([(x,0),(&l.gain,0),(&o.norm,0)]);
 if groups==0 {
  g.dispatch(e,"qmv_multi_g8_norm",&bufs,&[1,6144,2048,16,16,1024],[1,258,1],[256,1,1]);
  g.dispatch(e,"gdn_conv_prepare_decode",&[(&o.y[0],0),(&o.history,0),(&l.conv,0),(&o.conv,0),(&o.y[2],0),(&o.y[3],0),(&l.alog,0),(&l.dt,0),(&o.g,0),(&o.beta,0)],&[],[48,1,1],[32,1,1]);
 }else{
  bufs.extend([(&o.history,0),(&l.conv,0),(&o.conv,0),(&l.alog,0),(&l.dt,0),(&o.g,0),(&o.beta,0)]);
  g.dispatch(e,&format!("gdn_input_g{groups}_005"),&bufs,&[1,6144,2048,16,16,1024],[1,64,1],[groups*32,1,1]);
 }
}
fn compare(a:&Outputs,b:&Outputs)->(usize,usize){let mut d=0;let mut guard=0;for((_,a,n,width),(_,b,_,_))in a.all().into_iter().zip(b.all()){d+=bytes(a)[..n].chunks_exact(width).zip(bytes(b)[..n].chunks_exact(width)).filter(|(a,b)|a!=b).count();guard+=bytes(b)[n..].iter().filter(|v|**v!=0xa5).count();} (d,guard)}
fn main()->Result<()>{
 let args:Vec<String>=std::env::args().collect();let bench=args[1]=="bench";
 let mut g=Gpu::new()?;g.compile(include_str!("../src/kernels/linear_input_norm.metal"),&["qmv_multi_g8_norm"])?;g.compile(include_str!("../src/kernels/gdn.metal"),&["gdn_conv_prepare_decode"])?;
 g.compile(include_str!("gdn_input_005.metal"),&["gdn_input_g8_005","gdn_input_g16_005","gdn_input_g32_005"])?;
 let mut weights=Weights::load(&g,std::path::Path::new(&args[2]))?;let mut layers=vec![];
 for i in 0..24{if i%4==3{continue;}let p=format!("layers.{i}");let gain=weights.take(&format!("{p}.input_layernorm.weight"),&[1024],"BF16")?;let p=format!("{p}.linear_attn");let ws=[("in_proj_qkv",6144),("in_proj_z",2048),("in_proj_a",16),("in_proj_b",16)].iter().map(|(s,n)|weights.linear(&format!("{p}.{s}"),*n,1024)).collect::<Result<_>>()?;let conv=weights.take(&format!("{p}.conv1d.weight"),&[6144,4,1],"BF16")?;let alog=weights.take(&format!("{p}.A_log"),&[16],"F32")?;let dt=weights.take(&format!("{p}.dt_bias"),&[16],"BF16")?;layers.push(Layer{ws,gain,conv,alog,dt});}
 let mut seed=0x6134023098fau64;
 for (index,layer)in layers.iter().enumerate(){for scale in [0.0,0.001,0.1,1.0,10.0,100.0]{let outputs=(0..4).map(|_|Outputs::new(&g,&mut seed)).collect::<Result<Vec<_>>>()?;for o in &outputs[1..]{copy(&o.history,&outputs[0].history);}for step in 0..4{let x=g.from_bytes(&randbf(1024,&mut seed,scale))?;run(&g,1,|e|dispatch(&g,e,layer,&x,&outputs[0],0))?;for(i,groups)in [8,16,32].iter().enumerate(){run(&g,1,|e|dispatch(&g,e,layer,&x,&outputs[i+1],*groups))?;let(diff,guard_diff)=compare(&outputs[0],&outputs[i+1]);println!("{}",serde_json::json!({"check":"gdn_input","layer":index,"scale":scale,"step":step,"groups":groups,"diff":diff,"guard_diff":guard_diff}));ensure!(diff+guard_diff==0,"input fusion differs");}}}}
 if bench{let xs=(0..18).map(|_|g.from_bytes(&randbf(1024,&mut seed,1.0))).collect::<Result<Vec<_>>>()?;let os=(0..18).map(|_|Outputs::new(&g,&mut seed)).collect::<Result<Vec<_>>>()?;for groups in [0,8,16,32]{run(&g,8,|e|for(i,l)in layers.iter().enumerate(){dispatch(&g,e,l,&xs[i],&os[i],groups)})?;}for round in 0..24{let mut vars=vec![0,8,16,32];if round%2==1{vars.reverse();}for groups in vars{let us=run(&g,64,|e|for(i,l)in layers.iter().enumerate(){dispatch(&g,e,l,&xs[i],&os[i],groups)})?;println!("{}",serde_json::json!({"bench":"gdn_input_stream18","groups":groups,"round":round,"us":us}));}}}
 Ok(())
}
