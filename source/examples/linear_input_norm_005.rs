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
struct Layer { w:Vec<Linear>, gain:Buf }
struct Outputs{norm:Buf, y:Vec<Buf>}
impl Outputs{fn new(g:&Gpu)->Result<Self>{Ok(Self{norm:g.buffer(2112)?,y:[6144,2048,512,16].iter().map(|n|g.buffer(n*2+64)).collect::<Result<_>>()?})}}
fn poison(o:&Outputs){for b in std::iter::once(&o.norm).chain(o.y.iter()){unsafe{std::ptr::write_bytes(b.contents().as_ptr().cast::<u8>(),0xa5,b.length());}}}
fn guards(o:&Outputs,l:&Layer)->usize{let mut d=bytes(&o.norm)[2048..].iter().filter(|v|**v!=0xa5).count();for(i,b)in o.y.iter().enumerate(){let n=l.w.get(i).map_or(0,|w|w.n);d+=bytes(b)[n*2..].iter().filter(|v|**v!=0xa5).count();}d}
fn dispatch(g:&Gpu,e:&Enc,l:&Layer,x:&Buf,o:&Outputs,groups:usize,fused:bool){
 if !fused{g.dispatch(e,"rms_norm_bf16",&[(x,0),(&l.gain,0),(&o.norm,0)],&[1,1024,1e-6f32.to_bits()],[1,1,1],[256,1,1]);}
 let name=if groups==2 {"qmv_multi_q4".to_owned()}else{format!("qmv_multi_g{}{}",groups,if fused {"_norm"}else{""})};
 let mut bufs=vec![];for i in 0..4{let w=&l.w[i.min(l.w.len()-1)];bufs.extend([(&w.w,0),(&w.s,0),(&w.b,0),(&o.y[i],0)]);}bufs.push((if fused{x}else{&o.norm},0));if fused{bufs.push((&l.gain,0));bufs.push((&o.norm,0));}
 let ns=l.w.iter().map(|w|w.n).chain(std::iter::repeat(0)).take(4).collect::<Vec<_>>();let n=ns.iter().map(|n|n.div_ceil(groups*4)).sum();
 g.dispatch(e,&name,&bufs,&[1,ns[0]as u32,ns[1]as u32,ns[2]as u32,ns[3]as u32,1024],[1,n,1],[groups*32,1,1]);
}
fn main()->Result<()>{let args:Vec<String>=std::env::args().collect();let bench=args[1]=="bench";let mut g=Gpu::new()?;g.compile(include_str!("../src/kernels/linear.metal"),&["rms_norm_bf16","qmv_multi_q4"])?;
 let names=( [8,16,32].into_iter().flat_map(|g|[format!("qmv_multi_g{g}"),format!("qmv_multi_g{g}_norm")]) ).collect::<Vec<_>>();g.compile(include_str!("linear_input_norm_005.metal"),&names.iter().map(|x|x.as_str()).collect::<Vec<_>>())?;
 let mut w=Weights::load(&g,std::path::Path::new(&args[2]))?;let mut layers=vec![];for i in 0..24 {let p=format!("layers.{i}");let gain=w.take(&format!("{p}.input_layernorm.weight"),&[1024],"BF16")?;let specs=if i%4==3{vec![("self_attn.q_proj",4096),("self_attn.k_proj",512),("self_attn.v_proj",512)]}else{vec![("linear_attn.in_proj_qkv",6144),("linear_attn.in_proj_z",2048),("linear_attn.in_proj_a",16),("linear_attn.in_proj_b",16)]};let ws=specs.iter().map(|(name,n)|w.linear(&format!("{p}.{name}"),*n,1024)).collect::<Result<_>>()?;layers.push(Layer{w:ws,gain});}
 let mut seed=0x9af2deaba;let a=Outputs::new(&g)?;let b=Outputs::new(&g)?;let variants=vec![(2,false),(8,false),(8,true),(16,true),(32,true)];
 for trial in 0..if bench{1}else{4}{for scale in if bench{vec![1.0]}else{vec![0.0,0.001,0.1,1.0,10.0,100.0]}{let xs=(0..24).map(|_|g.from_bytes(&randbf(1024,&mut seed,scale))).collect::<Result<Vec<_>>>()?;
 for(i,l)in layers.iter().enumerate(){poison(&a);run(&g,1,|e|dispatch(&g,e,l,&xs[i],&a,2,false))?;ensure!(guards(&a,l)==0,"parent output guard changed");for(groups,fused)in &variants[1..]{poison(&b);run(&g,1,|e|dispatch(&g,e,l,&xs[i],&b,*groups,*fused))?;let gd=guards(&b,l);let nd=diffs(&a.norm,&b.norm,1024);let d=l.w.iter().enumerate().map(|(i,w)|diffs(&a.y[i],&b.y[i],w.n)).sum::<usize>();println!("{}",serde_json::json!({"check":"input_norm","layer":i,"trial":trial,"scale":scale,"groups":groups,"fused":fused,"diff":d,"norm_diff":nd,"guard_diff":gd}));ensure!(d+nd+gd==0,"input norm differs");}}
 if bench{for(groups,fused)in &variants{run(&g,8,|e|for(i,l)in layers.iter().enumerate(){dispatch(&g,e,l,&xs[i],&b,*groups,*fused)})?;}for round in 0..16{let mut vars=variants.clone();if round%2==1{vars.reverse();}for(groups,fused)in vars{let us=run(&g,128,|e|for(i,l)in layers.iter().enumerate(){dispatch(&g,e,l,&xs[i],&b,groups,fused)})?;println!("{}",serde_json::json!({"bench":"input_norm_stream24","groups":groups,"fused":fused,"round":round,"us":us}));}}}
 }}Ok(())}
