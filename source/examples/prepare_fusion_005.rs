#![allow(dead_code)]
#[path="../src/gpu.rs"] mod gpu;
use anyhow::{ensure, Result};
use gpu::{Buf, Gpu};
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandEncoder};

fn random(n:usize, seed:&mut u64, scale:f32)->Vec<u8> {
    let mut out=Vec::with_capacity(n*2);
    for _ in 0..n {
        *seed^=*seed<<13; *seed^=*seed>>7; *seed^=*seed<<17;
        let x=(((*seed>>40) as f32/(1u32<<23) as f32)-1.)*scale;
        out.extend_from_slice(&((x.to_bits()>>16) as u16).to_le_bytes());
    } out
}
fn bytes(b:&Buf)->&[u8] {
    unsafe {std::slice::from_raw_parts(b.contents().as_ptr().cast(),b.length())}
}
struct Input {q:Buf,k:Buf,v:Buf,qn:Buf,kn:Buf}
struct Output {q:Buf,k:Buf,v:Buf}
impl Output {
    fn new(g:&Gpu,c:usize)->Result<Self> {
        Ok(Self{q:g.buffer(2048*2)?,k:g.buffer(2*c*256*2)?,v:g.buffer(2*c*256*2)?})
    }
}
fn run(g:&Gpu,i:&Input,o:&Output,p:usize,c:usize,fused:bool,reps:usize)->Result<f64> {
    let params=[1,p as u32,c as u32];
    let (cmd,enc)=g.begin()?;
    for _ in 0..reps {
        if fused {
            g.dispatch(&enc,"attn_prepare_qkv_decode",&[(&i.q,0),(&i.k,0),(&i.v,0),(&i.qn,0),(&i.kn,0),(&o.q,0),(&o.k,0),(&o.v,0)],&params,[10,1,1],[64,1,1]);
        } else {
            g.dispatch(&enc,"attn_prepare_q",&[(&i.q,0),(&i.qn,0),(&o.q,0)],&params,[8,1,1],[64,1,1]);
            g.dispatch(&enc,"attn_prepare_kv",&[(&i.k,0),(&i.v,0),(&i.kn,0),(&o.k,0),(&o.v,0)],&params,[2,1,1],[64,1,1]);
        }
    }
    enc.endEncoding();cmd.commit();Gpu::wait(&cmd)?;
    Ok((cmd.GPUEndTime()-cmd.GPUStartTime())*1e6/reps as f64)
}
fn main()->Result<()> {
    let bench=std::env::args().nth(1).as_deref()==Some("bench");
    let mut g=Gpu::new()?;
    let source=include_str!("../src/kernels/attention.metal");
    g.compile(&source,&["attn_prepare_q","attn_prepare_kv","attn_prepare_qkv_decode"])?;
    let c=262144; let a=Output::new(&g,c)?;let b=Output::new(&g,c)?;
    for out in [&a,&b] {Gpu::zero(&out.q);Gpu::zero(&out.k);Gpu::zero(&out.v);}
    let mut cases=0;
    for pos in [0,1,31,1023,1024,4095,32767,32768,65536,131071,131328,262143] {
        for seed_id in 0..4 {for scale in [0.,0.001,1.,10.,100.] {
            let mut seed=0x1234587965413278u64+seed_id;
            let i=Input{q:g.from_bytes(&random(4096,&mut seed,scale))?,k:g.from_bytes(&random(512,&mut seed,scale))?,v:g.from_bytes(&random(512,&mut seed,scale))?,qn:g.from_bytes(&random(256,&mut seed,2.))?,kn:g.from_bytes(&random(256,&mut seed,2.))?};
            run(&g,&i,&a,pos,c,false,1)?;run(&g,&i,&b,pos,c,true,1)?;
            ensure!(bytes(&a.q)==bytes(&b.q),"query mismatch at {pos}/{seed_id}/{scale}");
            for h in 0..2 {
                let begin=(h*c+pos.saturating_sub(1))*512;
                let end=(h*c+(pos+2).min(c))*512;
                ensure!(bytes(&a.k)[begin..end]==bytes(&b.k)[begin..end],"key mismatch at {pos}/{seed_id}/{scale}");
                ensure!(bytes(&a.v)[begin..end]==bytes(&b.v)[begin..end],"value mismatch at {pos}/{seed_id}/{scale}");
            }
            cases+=1;
            if bench && pos==65536 && seed_id==0 && scale==1. {
                for round in 0..16 {
                    for fused in if round%2==0 {[false,true]}else{[true,false]} {
                        let us=run(&g,&i,&b,pos,c,fused,512)?;
                        println!("{{\"mode\":\"bench\",\"round\":{round},\"fused\":{fused},\"us\":{us}}}");
                    }
                }
            }
        }}
    }
    ensure!(bytes(&a.k)==bytes(&b.k)&&bytes(&a.v)==bytes(&b.v),"full cache mismatch");
    println!("{{\"mode\":\"check\",\"cases\":{cases},\"differences\":0}}");
    Ok(())
}
