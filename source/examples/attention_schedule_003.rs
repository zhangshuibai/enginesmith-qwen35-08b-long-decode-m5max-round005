#![allow(dead_code)]
#[path = "../src/gpu.rs"] mod gpu;
use anyhow::{bail, Result};
use gpu::{Buf, Gpu};
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandEncoder};

const VARIANTS: &[(&str,usize)] = &[("attn_decode_split1",4),
("schedule_p0_s1",1),
("schedule_p0_s2",2),
("schedule_p0_s4",4),
("schedule_p0_s8",8),
("schedule_p0_s16",16),
("schedule_p1_s1",1),
("schedule_p1_s2",2),
("schedule_p1_s4",4),
("schedule_p1_s8",8),
("schedule_p1_s16",16),
("schedule_p2_s1",1),
("schedule_p2_s2",2),
("schedule_p2_s4",4),
("schedule_p2_s8",8),
("schedule_p2_s16",16),
("schedule_p3_s1",1),
("schedule_p3_s2",2),
("schedule_p3_s4",4),
("schedule_p3_s8",8),
("schedule_p3_s16",16),
];
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
struct Inputs { q: Buf, k: Buf, v: Buf }
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
        g.dispatch(&enc,variant.0,&[(&inp.q,0),(&inp.k,0),(&inp.v,0),(&out.part,0),(&out.sums,0),(&out.maxs,0)],&p,if variant.0=="attn_decode_split1" {[2,1,64]}else{[512/variant.1,1,1]},[32,variant.1,1]);
        if combine {g.dispatch(&enc,"attn_decode_split2",&[(&out.part,0),(&out.sums,0),(&out.maxs,0),(&out.out,0)],&p,[8,1,1],[1024,1,1]);}
    }
    enc.endEncoding(); cmd.commit(); cmd.waitUntilCompleted();
    if let Some(e)=cmd.error() {bail!("GPU error: {e}");}
    Ok((cmd.GPUEndTime()-cmd.GPUStartTime())*1e6/reps as f64)
}

fn main()->Result<()> {
 let args:Vec<String>=std::env::args().collect();
 let timing=args.get(1).map(|s|s=="bench").unwrap_or(false);
 let lengths:Vec<usize>=args.get(2).map(|s|s.split(',').map(|v|v.parse().unwrap()).collect()).unwrap_or_else(||vec![1024,1025,4095,4096,32768,65536,131072]);
 let seeds:usize=args.get(3).map(|s|s.parse().unwrap()).unwrap_or(2);
 let mut g=Gpu::new()?;
 let source=[include_str!("../src/kernels/attention_decode.metal"),include_str!("attention_schedule_003.metal")].join("\n");
 let mut names:Vec<&str>=VARIANTS.iter().map(|v|v.0).collect();names.push("attn_decode_split2");g.compile(&source,&names)?;
 let reference=Outputs::new(&g)?;let candidate=Outputs::new(&g)?;
 let mut failures=0;
 for n in lengths {for si in 0..seeds {
   let mut seed=0x8726415512345678u64.wrapping_add(si as u64);
   let inp=Inputs {q:g.from_bytes(&random_bf16(8*256,&mut seed,if si%2==0{1.0}else{4.0}))?,k:g.from_bytes(&random_bf16(2*(n+32)*256,&mut seed,1.0))?,v:g.from_bytes(&random_bf16(2*(n+32)*256,&mut seed,1.0))?};
   run(&g,&inp,&reference,n,VARIANTS[0],1,true)?;
   for &variant in &VARIANTS[1..] {
     run(&g,&inp,&candidate,n,variant,1,true)?;
     let mut counts=Vec::new();
     for((label,a,width),(_,b,_)) in reference.all().into_iter().zip(candidate.all()) {
       let diffs=bytes(a).chunks_exact(width).zip(bytes(b).chunks_exact(width)).filter(|(a,b)|a!=b).count();failures+=diffs;counts.push(format!("\"{label}\":{diffs}"));
     }
     println!("{{\"mode\":\"check\",\"length\":{n},\"seed\":{si},\"variant\":\"{}\",{}}}",variant.0,counts.join(","));
   }
   if timing {for round in 0..4 {
     let indices:Vec<usize>=if round%2==0{(0..VARIANTS.len()).collect()}else{(0..VARIANTS.len()).rev().collect()};
     for i in indices {let variant=VARIANTS[i];let us=run(&g,&inp,&candidate,n,variant,16,false)?;println!("{{\"mode\":\"bench\",\"length\":{n},\"seed\":{si},\"round\":{round},\"variant\":\"{}\",\"us\":{us}}}",variant.0);}
   }}
 }}
 if failures>0 {bail!("{failures} element differences");} Ok(())
}
