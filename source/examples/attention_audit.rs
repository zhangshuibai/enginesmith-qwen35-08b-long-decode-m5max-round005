#![allow(dead_code)]
#[path = "../src/gpu.rs"] mod gpu;
use anyhow::{bail, Result};
use gpu::{Buf, Gpu};
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandEncoder};

const VARIANTS: &[(&str, usize)] = &[
    ("attn_decode_split1", 4),
    ("parent_tk1", 4), ("parent_tk2", 4), ("parent_tk8", 4),
    ("shared_h2_tk1", 2), ("shared_h2_tk2", 2),
    ("shared_h2_tk4", 2), ("shared_h2_tk8", 2),
    ("shared_h4_tk4", 1),
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
        g.dispatch(&enc,variant.0,&[(&inp.q,0),(&inp.k,0),(&inp.v,0),(&out.part,0),(&out.sums,0),(&out.maxs,0)],&p,[2,1,64],[32,variant.1,1]);
        if combine {g.dispatch(&enc,"attn_decode_split2",&[(&out.part,0),(&out.sums,0),(&out.maxs,0),(&out.out,0)],&p,[8,1,1],[1024,1,1]);}
    }
    enc.endEncoding(); cmd.commit(); cmd.waitUntilCompleted();
    if let Some(e)=cmd.error() {bail!("GPU error: {e}");}
    Ok((cmd.GPUEndTime()-cmd.GPUStartTime())*1e6/reps as f64)
}
fn run_stage2(g:&Gpu, inp:&Outputs, out:&Buf, n:usize, remap:bool, reps:usize)->Result<f64> {
    let (cmd,enc)=g.begin()?;
    for _ in 0..reps {
        g.dispatch(&enc,if remap {"split2_lanes"}else{"attn_decode_split2"},
            &[(&inp.part,0),(&inp.sums,0),(&inp.maxs,0),(out,0)],&[n as u32,(n+32) as u32],
            if remap {[8,32,1]}else{[8,1,1]},if remap {[32,1,1]}else{[1024,1,1]});
    }
    enc.endEncoding();cmd.commit();cmd.waitUntilCompleted();
    if let Some(e)=cmd.error() {bail!("GPU error: {e}");}
    Ok((cmd.GPUEndTime()-cmd.GPUStartTime())*1e6/reps as f64)
}
fn run_gated_stage2(g:&Gpu, inp:&Outputs, out:&Outputs, qp:&Buf, n:usize, variant:usize, reps:usize)->Result<f64> {
    let (cmd,enc)=g.begin()?;
    for _ in 0..reps {
        if variant>=2 {
            g.dispatch(&enc,if variant==2 {"split2_lanes_gate"}else{"attn_decode_split2_gate"},&[(&inp.part,0),(&inp.sums,0),(&inp.maxs,0),(&out.gated,0),(qp,0)],
                &[n as u32,(n+32) as u32],[8,32,1],[32,1,1]);
        } else {
            g.dispatch(&enc,if variant==1 {"split2_lanes"}else{"attn_decode_split2"},
                &[(&inp.part,0),(&inp.sums,0),(&inp.maxs,0),(&out.out,0)],&[n as u32,(n+32) as u32],
                if variant==1 {[8,32,1]}else{[8,1,1]},if variant==1 {[32,1,1]}else{[1024,1,1]});
            g.dispatch(&enc,"attn_gate",&[(&out.out,0),(qp,0),(&out.gated,0)],&[1],[8,1,1],[256,1,1]);
        }
    }
    enc.endEncoding();cmd.commit();cmd.waitUntilCompleted();
    if let Some(e)=cmd.error() {bail!("GPU error: {e}");}
    Ok((cmd.GPUEndTime()-cmd.GPUStartTime())*1e6/reps as f64)
}
fn main()->Result<()> {
    let args:Vec<String>=std::env::args().collect();
    let fusion_only=args.get(1).map(|x|x=="fused-bench").unwrap_or(false);
    let timing=args.get(1).map(|x|x=="bench" || x=="fused-bench").unwrap_or(false);
    let lengths:Vec<usize>=args.get(2).map(|s|s.split(',').map(|x|x.parse().unwrap()).collect())
        .unwrap_or_else(||vec![1024,1025,4095,4096,32768,65536,131072]);
    let seeds:usize=args.get(3).map(|s|s.parse().unwrap()).unwrap_or(2);
    let mut g=Gpu::new()?;
    let source=[include_str!("../src/kernels/attention_decode.metal"),include_str!("attention_audit.metal")].join("\n");
    let mut names:Vec<&str>=VARIANTS.iter().map(|x|x.0).collect(); names.extend(["attn_decode_split2","split2_lanes","split2_lanes_gate"]);
    g.compile(&source,&names)?;
    // Compile the shipped library separately, exactly as production does.
    // Keep the isolated variant above to retain the experiment's provenance.
    g.compile(include_str!("../src/kernels/attention_decode.metal"),&["attn_decode_split2_gate"])?;
    g.compile(include_str!("../src/kernels/attention.metal"),&["attn_gate"])?;
    let reference=Outputs::new(&g)?;let candidate=Outputs::new(&g)?;
    let mut failures=0;
    for n in lengths {
        for si in 0..seeds {
            let mut seed=0x987654321fedcba1u64.wrapping_add(si as u64);
            let q=g.from_bytes(&random_bf16(8*256,&mut seed,if si%2==0 {1.0}else{4.0}))?;
            let k=g.from_bytes(&random_bf16(2*(n+32)*256,&mut seed,1.0))?;
            let v=g.from_bytes(&random_bf16(2*(n+32)*256,&mut seed,1.0))?;
            let inp=Inputs{q,k,v};
            run(&g,&inp,&reference,n,VARIANTS[0],1,true)?;
            for &variant in &VARIANTS[1..] {
                run(&g,&inp,&candidate,n,variant,1,true)?;
                let mut counts=Vec::new();
                for ((label,a,width),(_,b,_)) in reference.all().into_iter().zip(candidate.all()) {
                    let diffs=bytes(a).chunks_exact(width).zip(bytes(b).chunks_exact(width)).filter(|(x,y)|x!=y).count();
                    failures+=diffs;
                    counts.push(format!("\"{label}\":{diffs}"));
                }
                println!("{{\"mode\":\"check\",\"length\":{n},\"seed\":{si},\"variant\":\"{}\",{}}}",variant.0,counts.join(","));
            }
            run_stage2(&g,&reference,&candidate.out,n,true,1)?;
            let diffs=bytes(&reference.out).chunks_exact(2).zip(bytes(&candidate.out).chunks_exact(2)).filter(|(x,y)|x!=y).count();
            failures+=diffs;
            println!("{{\"mode\":\"check\",\"length\":{n},\"seed\":{si},\"variant\":\"split2_lanes\",\"output\":{diffs}}}");
            for gate_scale in [0.01f32,1.0,8.0,120.0] {
                let mut gate_bytes=random_bf16(4096,&mut seed,gate_scale);
                // Explicit zero, signed zero, and large positive/negative gates.
                for (i,x) in [0.0f32,-0.0,88.0,-88.0,100.0,-100.0].iter().enumerate() {
                    let b=((x.to_bits()>>16) as u16).to_le_bytes();
                    gate_bytes[(256+i)*2..(256+i)*2+2].copy_from_slice(&b);
                }
                let qp=g.from_bytes(&gate_bytes)?;
                run_gated_stage2(&g,&reference,&reference,&qp,n,0,1)?;
                for variant in [2,3] {
                    run_gated_stage2(&g,&reference,&candidate,&qp,n,variant,1)?;
                    let diffs=bytes(&reference.gated).chunks_exact(2).zip(bytes(&candidate.gated).chunks_exact(2)).filter(|(x,y)|x!=y).count();
                    failures+=diffs;
                    let name=if variant==2 {"split2_lanes_gate"}else{"attn_decode_split2_gate"};
                    println!("{{\"mode\":\"check\",\"length\":{n},\"seed\":{si},\"variant\":\"{name}\",\"gate_scale\":{gate_scale},\"output\":{diffs}}}");
                }
            }
            if timing {
                if !fusion_only {for &variant in VARIANTS {run(&g,&inp,&candidate,n,variant,2,false)?;}}
                let qp=g.from_bytes(&random_bf16(4096,&mut seed,8.0))?;
                for variant in 0..3 {run_gated_stage2(&g,&reference,&candidate,&qp,n,variant,16)?;}
                for round in 0..4 {
                    let indices:Vec<usize>=if round%2==0 {(0..VARIANTS.len()).collect()}else{(0..VARIANTS.len()).rev().collect()};
                    for i in indices.into_iter().filter(|_|!fusion_only) {
                        let variant=VARIANTS[i];let us=run(&g,&inp,&candidate,n,variant,16,false)?;
                        println!("{{\"mode\":\"bench\",\"length\":{n},\"seed\":{si},\"round\":{round},\"variant\":\"{}\",\"us\":{us}}}",variant.0);
                    }
                    for remap in if round%2==0 {[false,true]}else{[true,false]} {
                        let us=run_stage2(&g,&reference,&candidate.out,n,remap,256)?;
                        let variant=if remap {"split2_lanes"}else{"split2_parent"};
                        println!("{{\"mode\":\"bench\",\"length\":{n},\"seed\":{si},\"round\":{round},\"variant\":\"{variant}\",\"us\":{us}}}");
                    }
                    for variant in if round%2==0 {[0,1,2]}else{[2,1,0]} {
                        let us=run_gated_stage2(&g,&reference,&candidate,&qp,n,variant,256)?;
                        let name=["split2_parent_gate","split2_remap_gate_separate","split2_lanes_gate"][variant];
                        println!("{{\"mode\":\"bench\",\"length\":{n},\"seed\":{si},\"round\":{round},\"variant\":\"{name}\",\"us\":{us}}}");
                    }
                }
            }
        }
    }
    if failures>0 {bail!("{failures} element differences");}
    Ok(())
}
