#![allow(dead_code)]
#[path = "../src/gpu.rs"] mod gpu;
use anyhow::{bail, Result};
use gpu::{Buf, Gpu};
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandEncoder};

const VARIANTS: &[(&str,usize)] = &[("attn_decode_split1",4),("scores_then_v2",4),("scores_then_v4",4),("scores_then_v8",4),("all_loads_then_scores2",4),("all_loads_then_scores4",4),("all_loads_then_scores8",4),("packed4",4),("packed8",4),("paired_exp4",4),("batch_exp4",4)];
fn random_bf16(n: usize, seed: &mut u64, scale: f32) -> Vec<u8> {
    let mut out = Vec::with_capacity(n * 2);
    for _ in 0..n {
        *seed ^= *seed << 13; *seed ^= *seed >> 7; *seed ^= *seed << 17;
        let x = (((*seed >> 40) as f32 / (1u32 << 23) as f32) - 1.0) * scale;
        out.extend_from_slice(&((x.to_bits() >> 16) as u16).to_le_bytes());
    }
    out
}
fn bytes(b: &Buf) -> &[u8] {
    unsafe { std::slice::from_raw_parts(b.contents().as_ptr().cast(), b.length()) }
}
struct Inputs { q: Buf, k: Buf, v: Buf, k_fold: Buf, v_fold: Buf }
struct Outputs { part: Buf, sums: Buf, maxs: Buf, out: Buf, gated: Buf }
impl Outputs {
    fn new(g: &Gpu) -> Result<Self> {
        Ok(Self { part:g.buffer(8*64*256*2)?, sums:g.buffer(8*64*4)?,
            maxs:g.buffer(8*64*4)?, out:g.buffer(8*256*2)?, gated:g.buffer(8*256*2)? })
    }
    fn all(&self) -> [(&str, &Buf, usize);4] {
        [("partial",&self.part,2),("sums",&self.sums,4),
         ("maxs",&self.maxs,4),("output",&self.out,2)]
    }
}
fn run(g:&Gpu, inp:&Inputs, out:&Outputs, n:usize, variant:(&str,usize), reps:usize, combine:bool) -> Result<f64> {
    let p=[n as u32,(n+32) as u32];
    let (cmd,enc)=g.begin()?;
    for _ in 0..reps {
        let (k,v)=if !variant.0.contains("_f1_") {(&inp.k,&inp.v)} else {(&inp.k_fold,&inp.v_fold)};
        g.dispatch(&enc,variant.0,&[(&inp.q,0),(k,0),(v,0),(&out.part,0),(&out.sums,0),(&out.maxs,0)],&p,[2,1,64],[32,variant.1,1]);
        if combine {g.dispatch(&enc,"attn_decode_split2",&[(&out.part,0),(&out.sums,0),(&out.maxs,0),(&out.out,0)],&p,[8,1,1],[1024,1,1]);}
    }
    enc.endEncoding(); cmd.commit(); cmd.waitUntilCompleted();
    if let Some(e)=cmd.error() {bail!("GPU error: {e}");}
    Ok((cmd.GPUEndTime()-cmd.GPUStartTime())*1e6/reps as f64)
}

fn fold(data:&[u8],capacity:usize)->Vec<u8> {
  let bc=capacity.div_ceil(64);
  let mut out=vec![0u8;2*64*bc*256*2];
  for h in 0..2 {for p in 0..capacity {
    let a=(h*capacity+p)*512;
    let b=((h*64+p%64)*bc+p/64)*512;
    out[b..b+512].copy_from_slice(&data[a..a+512]);
  }}
  out
}
fn main()->Result<()> {
 let args:Vec<String>=std::env::args().collect();
 let timing=args.get(1).map(|s|s=="bench").unwrap_or(false);
 let lengths:Vec<usize>=args.get(2).map(|s|s.split(',').map(|v|v.parse().unwrap()).collect()).unwrap_or_else(||vec![1024,1025,4095,4096,32768,65536,131072]);
 let seeds:usize=args.get(3).map(|s|s.parse().unwrap()).unwrap_or(2);
 let mut g=Gpu::new()?;
 g.compile(include_str!("attention_lifetime005.metal"),&["scores_then_v2","scores_then_v4","scores_then_v8","all_loads_then_scores2","all_loads_then_scores4","all_loads_then_scores8","packed4","packed8","paired_exp4","batch_exp4"])?;
 g.compile(include_str!("../src/kernels/attention_decode.metal"),&["attn_decode_split1","attn_decode_split2"])?;
 let reference=Outputs::new(&g)?;let candidate=Outputs::new(&g)?;
 let mut failures=0;
 for n in lengths {for si in 0..seeds {
   let mut seed=0x8726415512345678u64.wrapping_add(si as u64);
   let q=random_bf16(8*256,&mut seed,[0.0,1.0,4.0,16.0,64.0,0.00390625][si%6]);
   let k=random_bf16(2*(n+32)*256,&mut seed,1.0);
   let v=random_bf16(2*(n+32)*256,&mut seed,1.0);
   let inp=Inputs {q:g.from_bytes(&q)?,k:g.from_bytes(&k)?,v:g.from_bytes(&v)?,k_fold:g.buffer(4)?,v_fold:g.buffer(4)?};
   run(&g,&inp,&reference,n,VARIANTS[0],1,true)?;
   for &variant in &VARIANTS[1..] {
     if args.get(4).is_some_and(|s|s != variant.0) {continue;}
     run(&g,&inp,&candidate,n,variant,1,true)?;
     let mut counts=Vec::new();
     for((label,a,width),(_,b,_)) in reference.all().into_iter().zip(candidate.all()) {
       let diffs=bytes(a).chunks_exact(width).zip(bytes(b).chunks_exact(width)).filter(|(a,b)|a!=b).count();failures+=diffs;counts.push(format!("\"{label}\":{diffs}"));
     }
     println!("{{\"mode\":\"check\",\"length\":{n},\"seed\":{si},\"variant\":\"{}\",{}}}",variant.0,counts.join(","));
   }
   if timing {for round in 0..4 {
     let indices:Vec<usize>=if round%2==0{(0..VARIANTS.len()).collect()}else{(0..VARIANTS.len()).rev().collect()};
     for i in indices {let variant=VARIANTS[i];if i>0 && args.get(4).is_some_and(|s|s != variant.0) {continue;}let us=run(&g,&inp,&candidate,n,variant,16,false)?;println!("{{\"mode\":\"bench\",\"length\":{n},\"seed\":{si},\"round\":{round},\"variant\":\"{}\",\"us\":{us}}}",variant.0);}
   }}
 }}
 if failures>0 {bail!("{failures} element differences");} Ok(())
}
