#![allow(dead_code)]
#[path = "../src/gpu.rs"] mod gpu;
use anyhow::{ensure,Result};
use gpu::{Buf,Gpu};
use objc2_metal::{MTLBuffer, MTLCommandBuffer};
fn random(seed:&mut u64)->u64 { *seed^=*seed<<13; *seed^=*seed>>7; *seed^=*seed<<17; *seed }
fn read(b:&Buf)->u32 { unsafe{*b.contents().as_ptr().cast::<u32>()} }
fn run(g:&Gpu,x:&Buf,p:&Buf,b:&Buf,ids:&Buf,out:&Buf,count:usize,split:bool,reps:usize)->Result<f64> {
    let (cmd,enc)=g.begin()?;
    for _ in 0..reps {
        if split {
            g.dispatch(&enc,"greedy_partial",&[(x,0),(p,0)],&[count as u32],[32,1,1],[32,1,1]);
            g.dispatch(&enc,"greedy_candidates",&[(x,0),(p,0),(b,0),(ids,0)],&[count as u32],[32,1,1],[32,1,1]);
            g.dispatch(&enc,"greedy_finish",&[(b,0),(ids,0),(out,0)],&[],[1,1,1],[32,1,1]);
        } else {g.dispatch(&enc,"greedy_bf16",&[(x,0),(out,0)],&[count as u32],[1,1,1],[1024,1,1]);}
    }
    let cmd=Gpu::commit(cmd,enc);Gpu::wait(&cmd)?;
    Ok((cmd.GPUEndTime()-cmd.GPUStartTime())*1e6/reps as f64)
}
fn main()->Result<()> {
    let mut g=Gpu::new()?;
    g.compile(include_str!("../src/kernels/linear.metal"),&["greedy_bf16"])?;
    g.compile(include_str!("greedy_split_003.metal"),&["greedy_partial","greedy_candidates","greedy_finish"])?;
    let p=g.buffer(64*4)?;let b=g.buffer(32*4)?;let ids=g.buffer(32*4)?;
    let a=g.buffer(4)?;let c=g.buffer(4)?;
    let count=248320;let x=g.buffer(count*2)?;
    let mut seed=0x918beacd729au64;
    for si in 0..512 {
        let scale=[0.0f32,0.001,0.1,1.0,8.0,100.0,1000.0,30000.0][si%8];
        let offset=[0.0f32,-100.0,100.0,-10000.0][(si/8)%4];
        let ptr=x.contents().as_ptr().cast::<u16>();
        for i in 0..count {
            let v=((random(&mut seed)>>40) as f32/(1u32<<23) as f32-1.0)*scale+offset;
            unsafe{*ptr.add(i)=(v.to_bits()>>16) as u16;}
        }
        if si%7==0 { // Sparse maxima and ties scattered across reduction groups.
            for &i in &[0usize,31,32,1023,1024,1025,4095,4096,count-1] {
                unsafe{*ptr.add(i)=((offset+scale*1.01).to_bits()>>16) as u16;}
            }
        }
        run(&g,&x,&p,&b,&ids,&a,count,false,1)?;
        run(&g,&x,&p,&b,&ids,&c,count,true,1)?;
        ensure!(read(&a)==read(&c),"case {si}: parent {} split {}",read(&a),read(&c));
    }
    println!("{{\"mode\":\"check\",\"cases\":512,\"identical\":512}}");
    for round in 0..12 {
        for split in if round%2==0 {[false,true]}else{[true,false]} {
            let us=run(&g,&x,&p,&b,&ids,if split {&c}else{&a},count,split,128)?;
            println!("{{\"mode\":\"bench\",\"round\":{round},\"split\":{split},\"us\":{us}}}");
        }
    }
    Ok(())
}
