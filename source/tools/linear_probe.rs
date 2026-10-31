#![allow(dead_code)]
#[path = "../gpu.rs"]
mod gpu;
use anyhow::Result;
use gpu::Gpu;
use objc2_metal::MTLBuffer;
use std::path::Path;

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let dir = Path::new(&a[1]);
    let m: usize = a[2].parse()?;
    let n: usize = a[3].parse()?;
    let k: usize = a[4].parse()?;
    let mut g = Gpu::new()?;
    g.compile(include_str!("../kernels/linear.metal"), &["qmv_q4", "qmm_q4_plain", "embedding_q4", "rms_norm_bf16", "swiglu_bf16", "greedy_bf16"])?;
    g.compile(include_str!("../kernels/linear_nax.metal"), &["qmm_q4_nax"])?;
    g.compile(include_str!("../kernels/linear_splitk.metal"), &["qmm_q4_splitk", "qmm_q4_splitk_reduce"])?;
    let read = |name: &str| g.from_bytes(&std::fs::read(dir.join(name)).unwrap());
    let x = read("x.bin")?; let w = read("w.bin")?; let s = read("s.bin")?; let b = read("b.bin")?;
    let gain=read("gain.bin")?; let gate=read("gate.bin")?; let up=read("up.bin")?;
    let y=g.buffer(m*n*2)?; let r=g.buffer(m*k*2)?; let z=g.buffer(m*k*2)?;
    let save=|name:&str,buf:&gpu::Buf,bytes:usize|->Result<()> {
        let d=unsafe{std::slice::from_raw_parts(buf.contents().as_ptr().cast::<u8>(),bytes)};
        std::fs::write(dir.join(name),d)?;Ok(())
    };
    let ids:Vec<u32>=(0..m).map(|i|(i%n) as u32).collect();
    let ids=g.from_bytes(unsafe{std::slice::from_raw_parts(ids.as_ptr().cast::<u8>(),ids.len()*4)})?;
    let (cmd,enc)=g.begin()?;
    g.dispatch(&enc,"embedding_q4",&[(&w,0),(&s,0),(&b,0),(&ids,0),(&r,0)],&[m as u32,n as u32,k as u32],[((m*k)+255)/256,1,1],[256,1,1]);
    Gpu::finish(cmd,enc)?;save("embedding.bin",&r,m*k*2)?;
    if dir.join("logits.bin").exists() {
        let logits=read("logits.bin")?;let token=g.buffer(4)?;
        let (cmd,enc)=g.begin()?;
        g.dispatch(&enc,"greedy_bf16",&[(&logits,0),(&token,0)],&[(logits.length()/2) as u32],[1,1,1],[1024,1,1]);
        Gpu::finish(cmd,enc)?;save("token.bin",&token,4)?;
    }
    for name in ["qmv_q4", "qmm_q4_nax"] {
        let (cmd,enc)=g.begin()?;
        let (groups,threads)=if name=="qmv_q4" {([m,(n+7)/8,1],[64,1,1])} else {([(n+63)/64,(m+63)/64,1],[128,1,1])};
        g.dispatch(&enc,name,&[(&w,0),(&s,0),(&b,0),(&x,0),(&y,0)],&[m as u32,n as u32,k as u32],groups,threads);
        Gpu::finish(cmd,enc)?;save(&format!("{name}.bin"),&y,m*n*2)?;
    }
    let mut splits=(512/(((m+31)/32)*((n+31)/32))).max(1).min(k/64);
    while splits>1 && k%(splits*64)!=0 {splits-=1;}
    let partial=g.buffer(splits*m*n*4)?;
    let p=[m as u32,n as u32,k as u32,splits as u32];
    let (cmd,enc)=g.begin()?;
    g.dispatch(&enc,"qmm_q4_splitk",&[(&w,0),(&s,0),(&b,0),(&x,0),(&partial,0)],&p,[(n+31)/32,(m+31)/32,splits],[128,1,1]);
    g.dispatch(&enc,"qmm_q4_splitk_reduce",&[(&partial,0),(&y,0)],&p,[((m*n)+255)/256,1,1],[256,1,1]);
    Gpu::finish(cmd,enc)?;save("qmm_q4_splitk.bin",&y,m*n*2)?;
    let (cmd,enc)=g.begin()?;
    g.dispatch(&enc,"rms_norm_bf16",&[(&x,0),(&gain,0),(&r,0)],&[m as u32,k as u32,1e-6f32.to_bits()],[m,1,1],[((k+127)/128)*32,1,1]);
    g.dispatch(&enc,"swiglu_bf16",&[(&gate,0),(&up,0),(&z,0)],&[(m*k) as u32],[((m*k)+255)/256,1,1],[256,1,1]);
    Gpu::finish(cmd,enc)?;
    save("rms.bin",&r,m*k*2)?;save("swiglu.bin",&z,m*k*2)?;
    Ok(())
}
