//! Development-only byte comparison and isolated timing of the decode fusion.
//! The caller must hold /tmp/engine-gpu.lock for the lifetime of this process.
#![allow(dead_code)]
#[path = "../src/gpu.rs"]
mod gpu;

use anyhow::{ensure, Result};
use gpu::{Buf, Enc, Gpu};
use objc2_metal::{MTLBuffer, MTLCommandBuffer};

fn bytes(b: &Buf) -> &[u8] {
    unsafe { std::slice::from_raw_parts(b.contents().as_ptr().cast(), b.length()) }
}
fn copy(dst: &Buf, src: &Buf) {
    assert_eq!(dst.length(), src.length());
    unsafe { std::ptr::copy_nonoverlapping(src.contents().as_ptr(), dst.contents().as_ptr(), src.length()); }
}
fn random(seed: &mut u64) -> f32 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    ((*seed >> 40) as f32 / (1u32 << 23) as f32) - 1.0
}
fn bf16(g: &Gpu, n: usize, seed: &mut u64, scale: f32) -> Result<Buf> {
    let mut data = Vec::with_capacity(n * 2);
    for i in 0..n {
        let x = if i < 8 { [0.0, -0.0, 1.0, -1.0, 0.00001, -0.00001, 88.0, -88.0][i] * scale }
            else { random(seed) * scale };
        data.extend_from_slice(&((x.to_bits() >> 16) as u16).to_le_bytes());
    }
    g.from_bytes(&data)
}
fn f32s(g: &Gpu, n: usize, seed: &mut u64, scale: f32) -> Result<Buf> {
    let data: Vec<u8> = (0..n).flat_map(|_| (random(seed) * scale).to_le_bytes()).collect();
    g.from_bytes(&data)
}
struct Inputs {
    raw: Buf, weight: Buf, a: Buf, b: Buf, alog: Buf, dt: Buf, z: Buf, norm: Buf,
}
impl Inputs {
    fn new(g: &Gpu, seed: &mut u64, scale: f32) -> Result<Self> {
        Ok(Self {
            raw: bf16(g, 6144, seed, scale)?, weight: bf16(g, 6144 * 4, seed, 0.2)?,
            a: bf16(g, 16, seed, scale)?, b: bf16(g, 16, seed, scale)?,
            alog: f32s(g, 16, seed, 2.0)?, dt: bf16(g, 16, seed, 0.3)?,
            z: bf16(g, 2048, seed, scale)?, norm: bf16(g, 128, seed, 1.0)?,
        })
    }
}
struct Outputs {
    history: Buf, conv: Buf, g: Buf, beta: Buf, state: Buf, y: Buf, out: Buf,
}
impl Outputs {
    fn new(g: &Gpu, seed: &mut u64) -> Result<Self> {
        Ok(Self {
            history: bf16(g, 3 * 6144, seed, 1.0)?, conv: g.buffer(6144 * 2)?,
            g: g.buffer(16 * 4)?, beta: g.buffer(16 * 2)?,
            state: f32s(g, 16 * 128 * 128, seed, 0.1)?,
            y: g.buffer(2048 * 2)?, out: g.buffer(2048 * 2)?,
        })
    }
    fn reset_from(&self, src: &Self) {
        copy(&self.history, &src.history);
        copy(&self.state, &src.state);
    }
    fn all(&self) -> [(&str, &Buf, usize); 7] {
        [("history", &self.history, 2), ("qkv", &self.conv, 2), ("g", &self.g, 4),
         ("beta", &self.beta, 2), ("state", &self.state, 4), ("y", &self.y, 2),
         ("out", &self.out, 2)]
    }
}
fn encode(g: &Gpu, enc: &Enc, inp: &Inputs, out: &Outputs, variant: usize, prep: bool) {
    if prep {g.dispatch(enc, "gdn_conv_prepare_decode",
        &[(&inp.raw,0), (&out.history,0), (&inp.weight,0), (&out.conv,0),
          (&inp.a,0), (&inp.b,0), (&inp.alog,0), (&inp.dt,0), (&out.g,0), (&out.beta,0)],
        &[], [48,1,1], [32,1,1]);}
    let sg = [2,1,2,4,8,16,1,4,8,16][variant];
    let name = if (1..=5).contains(&variant) { "gdn_recur_vec005" } else { "gdn_recur" };
    g.dispatch(enc, name, &[(&out.conv,0), (&out.g,0), (&out.beta,0), (&out.state,0), (&out.y,0)],
        &[1], [1,16/sg,16], [32,sg,1]);
    g.dispatch(enc, "gdn_norm_gate", &[(&out.y,0), (&inp.z,0), (&inp.norm,0), (&out.out,0)],
        &[1], [16,1,1], [32,1,1]);
}
fn run(g: &Gpu, inp: &Inputs, out: &Outputs, variant: usize, reps: usize, prep: bool) -> Result<f64> {
    let (cmd, enc) = g.begin()?;
    for _ in 0..reps { encode(g, &enc, inp, out, variant, prep); }
    let cmd = Gpu::commit(cmd, enc); Gpu::wait(&cmd)?;
    Ok((cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e6 / reps as f64)
}
fn compare(a: &Outputs, b: &Outputs, variant: usize, case: usize, step: usize) -> usize {
    let mut failed = 0; let mut counts = Vec::new();
    for ((name, x, width), (_, y, _)) in a.all().into_iter().zip(b.all()) {
        let diff = bytes(x).chunks_exact(width).zip(bytes(y).chunks_exact(width)).filter(|(x,y)| x != y).count();
        failed += diff; counts.push(format!("\"{name}\":{diff}"));
    }
    println!("{{\"mode\":\"check\",\"variant\":{variant},\"case\":{case},\"step\":{step},{}}}", counts.join(",")); failed
}
fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect(); let bench = args.get(1).map(|s| s == "bench").unwrap_or(false);
    let seeds: usize = args.get(2).map(|s| s.parse()).transpose()?.unwrap_or(8);
    let mut gpu = Gpu::new()?;
    gpu.compile(include_str!("../src/kernels/gdn.metal"), &["gdn_conv_prepare_decode", "gdn_recur", "gdn_norm_gate"])?;
    gpu.compile(include_str!("gdn_vector_005.metal"), &["gdn_recur_vec005"])?;
    for si in 0..seeds { for (scale_i, scale) in [0.0, 0.01, 1.0, 8.0, 120.0].into_iter().enumerate() {
        let case = si*5+scale_i; let mut seed = 0x1edc_ba987_654321u64.wrapping_add(si as u64);
        let initial = Outputs::new(&gpu, &mut seed)?;
        let variants: Vec<Outputs> = (0..10).map(|_|Outputs::new(&gpu,&mut seed)).collect::<Result<_>>()?;
        for out in &variants {out.reset_from(&initial);}
        for step in 0..8 {
            let inp = Inputs::new(&gpu,&mut seed,scale)?;
            for (variant,out) in variants.iter().enumerate() {
                run(&gpu,&inp,out,variant,1,true)?;
                if variant>0 {ensure!(compare(&variants[0],out,variant,case,step)==0,"different outputs");}
            }
        }
    }}
    if bench {
        let mut seed=0x128497126bf; let inp=Inputs::new(&gpu,&mut seed,1.0)?;
        let out=Outputs::new(&gpu,&mut seed)?;run(&gpu,&inp,&out,0,1,true)?;
        for variant in 0..10 {run(&gpu,&inp,&out,variant,32,false)?;}
        for round in 0..32 {let mut variants:Vec<_>=(0..10).collect();if round%2==1 {variants.reverse();}
            for variant in variants {let us=run(&gpu,&inp,&out,variant,256,false)?;
                println!("{}",serde_json::json!({"mode":"bench","variant":variant,"round":round,"us":us}));}
        }
    }
    Ok(())
}
