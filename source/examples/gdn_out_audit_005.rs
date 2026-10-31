//! Audit the production GDN norm/output fusion against the original kernels.
//! Usage: gdn_out_audit_005 check|bench MODEL_DIR, under the shared GPU lock.
#![allow(dead_code)]
#[path = "../src/gpu.rs"]
mod gpu;
#[path = "../src/weights.rs"]
mod weights;

use anyhow::{ensure, Result};
use gpu::{Buf, Enc, Gpu};
use objc2_metal::{MTLBuffer, MTLCommandBuffer};
use weights::{Linear, Weights};

fn next(s: &mut u64) -> u64 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    *s
}

fn randbf(n: usize, seed: &mut u64, scale: f32) -> Vec<u8> {
    let mut values = Vec::with_capacity(n * 2);
    for i in 0..n {
        let value = (((next(seed) >> 40) as f32 / (1u32 << 23) as f32) - 1.0) * scale;
        let bits = match i % 251 {
            0 => 0u16,
            1 => 0x8000,
            _ => (value.to_bits() >> 16) as u16,
        };
        values.extend_from_slice(&bits.to_le_bytes());
    }
    values
}

fn bytes(buffer: &Buf) -> &[u8] {
    unsafe { std::slice::from_raw_parts(buffer.contents().as_ptr().cast(), buffer.length()) }
}

fn diffs(a: &Buf, b: &Buf, n: usize) -> usize {
    bytes(a)[..n * 2].chunks_exact(2)
        .zip(bytes(b)[..n * 2].chunks_exact(2))
        .filter(|(a, b)| a != b).count()
}

fn run<F: FnMut(&Enc)>(gpu: &Gpu, reps: usize, mut encode: F) -> Result<f64> {
    let (command, encoder) = gpu.begin()?;
    for _ in 0..reps { encode(&encoder); }
    let command = Gpu::commit(command, encoder);
    Gpu::wait(&command)?;
    Ok((command.GPUEndTime() - command.GPUStartTime()) * 1e6 / reps as f64)
}

struct Outputs { norm: Buf, out: Buf }
struct Layer { out: Linear, gain: Buf, y: Buf, z: Buf, residual: Buf }

fn dispatch(gpu: &Gpu, encoder: &Enc, layer: &Layer, output: &Outputs, fused: bool, alias: bool) {
    let residual = if alias { &output.out } else { &layer.residual };
    if fused {
        gpu.dispatch(encoder, "gdn_out_g8", &[
            (&layer.out.w, 0), (&layer.out.s, 0), (&layer.out.b, 0),
            (&layer.y, 0), (&layer.z, 0), (&layer.gain, 0),
            (residual, 0), (&output.out, 0), (&output.norm, 0),
        ], &[1, 1024, 2048], [1, 32, 1], [256, 1, 1]);
    } else {
        gpu.dispatch(encoder, "gdn_norm_gate", &[
            (&layer.y, 0), (&layer.z, 0), (&layer.gain, 0), (&output.norm, 0),
        ], &[1], [16, 1, 1], [32, 1, 1]);
        gpu.dispatch(encoder, "qmv_add_q4", &[
            (&layer.out.w, 0), (&layer.out.s, 0), (&layer.out.b, 0),
            (&output.norm, 0), (residual, 0), (&output.out, 0),
        ], &[1, 1024, 2048], [1, 128, 1], [64, 1, 1]);
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    ensure!(args.len() == 3, "usage: gdn_out_audit_005 check|bench MODEL_DIR");
    let bench = args[1] == "bench";
    let mut gpu = Gpu::new()?;
    gpu.compile(include_str!("../src/kernels/linear.metal"), &["qmv_add_q4"])?;
    gpu.compile(include_str!("../src/kernels/gdn.metal"), &["gdn_norm_gate"])?;
    gpu.compile(include_str!("gdn_out_rejected_005.metal"), &["gdn_out_g8"])?;
    let mut weights = Weights::load(&gpu, std::path::Path::new(&args[2]))?;
    let mut seed = 0x9187231afed34;
    let parent = Outputs { norm: gpu.buffer(4096)?, out: gpu.buffer(2048)? };
    let candidate = Outputs { norm: gpu.buffer(4096)?, out: gpu.buffer(2048)? };
    let mut layers = Vec::new();

    for index in 0..24 {
        if index % 4 == 3 { continue; }
        let out = weights.linear(&format!("layers.{index}.linear_attn.out_proj"), 1024, 2048)?;
        let gain = weights.take(&format!("layers.{index}.linear_attn.norm.weight"), &[128], "BF16")?;
        let mut layer = Layer {
            out, gain, y: gpu.buffer(4096)?, z: gpu.buffer(4096)?,
            residual: gpu.buffer(2048)?,
        };
        for scale in [0.0, 0.001, 0.1, 1.0, 10.0, 100.0] {
            layer.y = gpu.from_bytes(&randbf(2048, &mut seed, scale))?;
            layer.z = gpu.from_bytes(&randbf(2048, &mut seed, scale))?;
            layer.residual = gpu.from_bytes(&randbf(1024, &mut seed, scale))?;
            run(&gpu, 1, |encoder| dispatch(&gpu, encoder, &layer, &parent, false, false))?;
            run(&gpu, 1, |encoder| dispatch(&gpu, encoder, &layer, &candidate, true, false))?;
            let out_diff = diffs(&parent.out, &candidate.out, 1024);
            let norm_diff = diffs(&parent.norm, &candidate.norm, 2048);
            unsafe {
                std::ptr::copy_nonoverlapping(layer.residual.contents().as_ptr(),
                    candidate.out.contents().as_ptr(), layer.residual.length());
            }
            run(&gpu, 1, |encoder| dispatch(&gpu, encoder, &layer, &candidate, true, true))?;
            let alias_out_diff = diffs(&parent.out, &candidate.out, 1024);
            let alias_norm_diff = diffs(&parent.norm, &candidate.norm, 2048);
            println!("{}", serde_json::json!({
                "check": "gdn_out", "layer": index, "scale": scale,
                "groups": 8, "out_diff": out_diff, "norm_diff": norm_diff,
                "alias_out_diff": alias_out_diff, "alias_norm_diff": alias_norm_diff,
            }));
            ensure!(out_diff + norm_diff + alias_out_diff + alias_norm_diff == 0, "gdn norm+output differs");
        }
        layers.push(layer);
    }

    if bench {
        for fused in [false, true] {
            run(&gpu, 32, |encoder| dispatch(&gpu, encoder, &layers[0], &candidate, fused, false))?;
        }
        // The 18-layer cycle is larger than the hot-layer test, but still fits
        // SLC. These timings are diagnostic; normal-protocol A/B is required.
        for stream in [false, true] {
            for round in 0..24 {
                let variants = if round % 2 == 0 { [false, true] } else { [true, false] };
                for fused in variants {
                    let us = run(&gpu, if stream { 32 } else { 256 }, |encoder| {
                        if stream {
                            for layer in &layers { dispatch(&gpu, encoder, layer, &candidate, fused, false); }
                        } else {
                            dispatch(&gpu, encoder, &layers[0], &candidate, fused, false);
                        }
                    })?;
                    println!("{}", serde_json::json!({
                        "bench": "gdn_out", "stream": stream, "groups": if fused { 8 } else { 0 },
                        "round": round, "us": us,
                    }));
                }
            }
        }
    }
    Ok(())
}
