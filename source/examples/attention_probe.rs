#![allow(dead_code)]
#[path="../src/gpu.rs"] mod gpu;
use anyhow::Result;
use gpu::Gpu;
use objc2_metal::{MTLBuffer,MTLDevice};
use std::path::Path;
fn main()->Result<()> {
    let args:Vec<String>=std::env::args().collect();
    let dir=Path::new(&args[1]);let t:usize=args[2].parse()?;let n:usize=args[3].parse()?;
    let c=n+32;let offset=n-t;
    let mut g=Gpu::new()?;
    eprintln!("architecture {:?}",g.device.architecture().name());
    g.compile(include_str!("../src/kernels/attention.metal"),&["attn_prepare_q","attn_prepare_kv","attn_gate","attn_short_scores","attn_short_softmax","attn_short_values"])?;
    g.compile(include_str!("../src/kernels/attention_decode.metal"),&["attn_decode","attn_decode_split1","attn_decode_split2","attn_small"])?;
    g.compile(include_str!("../src/kernels/attention_prefill.metal"),&["attn_prefill"])?;
    let read=|s:&str|g.from_bytes(&std::fs::read(dir.join(format!("{s}.bin"))).unwrap());
    let qp=read("qp")?;let kp=read("kp")?;let vp=read("vp")?;
    let qn=read("qn")?;let kn=read("kn")?;let k=read("kc")?;let v=read("vc")?;
    let q=g.buffer(t*2048*2)?;let out=g.buffer(t*2048*2)?;let gate=g.buffer(t*2048*2)?;
    let partial=g.buffer(8*64*256*2)?;let sums=g.buffer(8*64*4)?;let maxs=g.buffer(8*64*4)?;
    let (cmd,enc)=g.begin()?;
    let p=[t as u32,offset as u32,c as u32];
    g.dispatch(&enc,"attn_prepare_q",&[(&qp,0),(&qn,0),(&q,0)],&p,[8,t,1],[64,1,1]);
    g.dispatch(&enc,"attn_prepare_kv",&[(&kp,0),(&vp,0),(&kn,0),(&k,0),(&v,0)],&p,[2,t,1],[64,1,1]);
    if t==1 {
        let p=[n as u32,c as u32];
        if n<1024 {g.dispatch(&enc,"attn_decode",&[(&q,0),(&k,0),(&v,0),(&out,0)],&p,[8,1,1],[1024,1,1]);}
        else {
            g.dispatch(&enc,"attn_decode_split1",&[(&q,0),(&k,0),(&v,0),(&partial,0),(&sums,0),(&maxs,0)],&p,[2,1,64],[32,4,1]);
            g.dispatch(&enc,"attn_decode_split2",&[(&partial,0),(&sums,0),(&maxs,0),(&out,0)],&p,[8,1,1],[1024,1,1]);
        }
    } else if t<=8 && n<1024 {
        g.dispatch(&enc,"attn_small",&[(&q,0),(&k,0),(&v,0),(&out,0)],&[n as u32,c as u32],[8,t,1],[1024,1,1]);
    } else if t>8 && t<1024 && n<=4096 {
        let scores=g.buffer(t*8*n*2)?;
        let p=[t as u32,n as u32,c as u32,offset as u32];
        g.dispatch(&enc,"attn_short_scores",&[(&q,0),(&k,0),(&scores,0)],&p,[n.div_ceil(4),8,t],[128,1,1]);
        g.dispatch(&enc,"attn_short_softmax",&[(&scores,0)],&p,[t*8,1,1],[n.div_ceil(128)*32,1,1]);
        g.dispatch(&enc,"attn_short_values",&[(&scores,0),(&v,0),(&out,0)],&p,[8,t,1],[256,1,1]);
    } else {
        let mut p=vec![1,8,256,t as u32,n as u32,4,(1f32/16.).to_bits(),t.div_ceil(64) as u32,n.div_ceil(32) as u32,(t/64) as u32,(n/32) as u32,(t%64) as u32,(n%32) as u32,offset as u32];
        for s in [t*2048,256,2048,2*c*256,c*256,256,2*c*256,c*256,256,t*2048,256,2048] {p.push(s as u32);p.push((s as u64>>32) as u32);}
        g.dispatch(&enc,"attn_prefill",&[(&q,0),(&k,0),(&v,0),(&out,0)],&p,[t.div_ceil(64),8,1],[256,1,1]);
    }
    g.dispatch(&enc,"attn_gate",&[(&out,0),(&qp,0),(&gate,0)],&[t as u32],[(t*2048).div_ceil(256),1,1],[256,1,1]);
    Gpu::finish(cmd,enc)?;
    for (name,b) in [("q",&q),("k",&k),("v",&v),("out",&out),("gate",&gate)] {
        let bytes=unsafe{std::slice::from_raw_parts(b.contents().as_ptr().cast::<u8>(),b.length())};std::fs::write(dir.join(format!("{name}.bin")),bytes)?;
    }
    Ok(())
}
