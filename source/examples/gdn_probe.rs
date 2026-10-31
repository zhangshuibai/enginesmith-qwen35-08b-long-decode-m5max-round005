//! Development-only stage comparison. Run through tools/probe_gdn.py so the
//! reference and native kernels share the process-level GPU lock.
#[allow(dead_code)]
#[path = "../src/gpu.rs"]
mod gpu;

use anyhow::{ensure, Result};
use gpu::{Buf, Gpu};
use objc2_metal::MTLBuffer;
use std::{fs, path::Path};

fn read(gpu: &Gpu, dir: &Path, name: &str, bytes: usize) -> Result<Buf> {
    let data = fs::read(dir.join(format!("{name}.bin")))?;
    ensure!(data.len() == bytes, "fixture {name}: wrong size");
    gpu.from_bytes(&data)
}
fn save(dir: &Path, name: &str, b: &Buf) -> Result<()> {
    let data = unsafe { std::slice::from_raw_parts(b.contents().as_ptr().cast::<u8>(), b.length()) };
    fs::write(dir.join(format!("native_{name}.bin")), data)?;
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    ensure!(args.len() == 3, "usage: gdn_probe FIXTURE_DIR ROWS");
    let dir = Path::new(&args[1]);
    let rows: usize = args[2].parse()?;
    let mut gpu = Gpu::new()?;
    gpu.compile(include_str!("../src/kernels/gdn.metal"), &[
        "gdn_conv", "gdn_history", "gdn_prepare", "gdn_recur", "gdn_norm_gate"
    ])?;
    let raw = read(&gpu, dir, "raw", rows * 6144 * 2)?;
    let history = read(&gpu, dir, "history", 3 * 6144 * 2)?;
    let conv_w = read(&gpu, dir, "conv_weight", 6144 * 4 * 2)?;
    let a = read(&gpu, dir, "a", rows * 16 * 2)?;
    let b = read(&gpu, dir, "b", rows * 16 * 2)?;
    let alog = read(&gpu, dir, "alog", 16 * 4)?;
    let dt = read(&gpu, dir, "dt", 16 * 2)?;
    let state = read(&gpu, dir, "state", 16 * 128 * 128 * 4)?;
    let z = read(&gpu, dir, "z", rows * 2048 * 2)?;
    let norm = read(&gpu, dir, "norm", 128 * 2)?;
    let conv = gpu.buffer(rows * 6144 * 2)?;
    let g = gpu.buffer(rows * 16 * 4)?;
    let beta = gpu.buffer(rows * 16 * 2)?;
    let y = gpu.buffer(rows * 2048 * 2)?;
    let out = gpu.buffer(rows * 2048 * 2)?;
    let p = [rows as u32];

    let (cmd, enc) = gpu.begin()?;
    gpu.dispatch(&enc, "gdn_conv", &[(&raw,0),(&history,0),(&conv_w,0),(&conv,0)], &p,
        [(rows*6144).div_ceil(256),1,1], [256,1,1]);
    gpu.dispatch(&enc, "gdn_history", &[(&raw,0),(&history,0)], &p, [24,1,1], [256,1,1]);
    Gpu::finish(cmd,enc)?;
    save(dir,"conv",&conv)?;
    save(dir,"history",&history)?;

    let (cmd, enc) = gpu.begin()?;
    gpu.dispatch(&enc, "gdn_prepare", &[(&conv,0),(&a,0),(&b,0),(&alog,0),(&dt,0),(&g,0),(&beta,0)], &p,
        [rows*16,1,1], [32,1,1]);
    Gpu::finish(cmd,enc)?;
    save(dir,"qkv",&conv)?;
    save(dir,"g",&g)?;
    save(dir,"beta",&beta)?;

    let (cmd, enc) = gpu.begin()?;
    gpu.dispatch(&enc,"gdn_recur",&[(&conv,0),(&g,0),(&beta,0),(&state,0),(&y,0)],&p,[1,8,16],[32,2,1]);
    gpu.dispatch(&enc,"gdn_norm_gate",&[(&y,0),(&z,0),(&norm,0),(&out,0)],&p,[rows*16,1,1],[32,1,1]);
    Gpu::finish(cmd,enc)?;
    save(dir,"state",&state)?;
    save(dir,"y",&y)?;
    save(dir,"out",&out)?;
    Ok(())
}
