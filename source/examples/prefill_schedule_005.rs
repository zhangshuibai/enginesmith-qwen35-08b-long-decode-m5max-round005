#![allow(dead_code)]
#[path="../src/gpu.rs"] mod gpu;
use anyhow::{bail,Result};
use gpu::{Buf,Gpu};
use objc2_metal::{MTLBuffer,MTLCommandBuffer,MTLCommandEncoder};
const NAMES:&[&str]=&["attn_prefill","schedule_parent","head_fast","gqa_fast","kv_interleave","reverse_query"];
fn random(n:usize,seed:&mut u64,scale:f32)->Vec<u8>{let mut r=Vec::with_capacity(n*2);for _ in 0..n{*seed^=*seed<<13;*seed^=*seed>>7;*seed^=*seed<<17;let x=(((*seed>>40)as f32/(1u32<<23)as f32)-1.)*scale;r.extend_from_slice(&((x.to_bits()>>16)as u16).to_le_bytes());}r}
fn bytes(b:&Buf)->&[u8]{unsafe{std::slice::from_raw_parts(b.contents().as_ptr().cast(),b.length())}}
fn cache(g:&Gpu,n:usize,seed:&mut u64)->Result<Buf>{
 let c=262144;let b=g.buffer(2*c*256*2)?;
 for head in 0..2 {let data=random((n+32)*256,seed,1.);unsafe{std::ptr::copy_nonoverlapping(data.as_ptr(),b.contents().as_ptr().cast::<u8>().add(head*c*256*2),data.len());}}
 Ok(b)
}
fn run(g:&Gpu,q:&Buf,k:&Buf,v:&Buf,out:&Buf,t:usize,n:usize,name:&str,reps:usize)->Result<f64>{
 let c=262144;let offset=n-t;
 let mut p=vec![1,8,256,t as u32,n as u32,4,(1f32/16.).to_bits(),t.div_ceil(64)as u32,n.div_ceil(32)as u32,(t/64)as u32,(n/32)as u32,(t%64)as u32,(n%32)as u32,offset as u32];
 for s in [t*2048,256,2048,2*c*256,c*256,256,2*c*256,c*256,256,t*2048,256,2048]{p.push(s as u32);p.push((s as u64>>32)as u32);}
 let(cmd,enc)=g.begin()?;
 for _ in 0..reps{g.dispatch(&enc,name,&[(q,0),(k,0),(v,0),(out,0)],&p,[t.div_ceil(64),8,1],[256,1,1]);}
 enc.endEncoding();cmd.commit();cmd.waitUntilCompleted();if let Some(e)=cmd.error(){bail!("GPU error {e}");}Ok((cmd.GPUEndTime()-cmd.GPUStartTime())*1e6/reps as f64)
}
fn main()->Result<()>{
 let args:Vec<String>=std::env::args().collect();let bench=args.get(1).is_some_and(|s|s=="bench");let seeds:usize=args.get(3).map(|s|s.parse().unwrap()).unwrap_or(2);
 let cases:Vec<(usize,usize)>=args.get(2).unwrap_or(&"2048:2048,2048:32768,2048:65536,2048:131072".into()).split(',').map(|x|{let(t,n)=x.split_once(':').unwrap();(t.parse().unwrap(),n.parse().unwrap())}).collect();
 let mut g=Gpu::new()?;g.compile(include_str!("../src/kernels/attention_prefill.metal"),&[NAMES[0]])?;g.compile(include_str!("prefill_schedule_005.metal"),&NAMES[1..])?;let mut failures=0;
 for(t,n)in cases{for si in 0..seeds{let mut seed=0x76543abcdef12348u64.wrapping_add(si as u64);let q=g.from_bytes(&random(t*2048,&mut seed,[1.,4.,0.][si%3]))?;let k=cache(&g,n,&mut seed)?;let v=cache(&g,n,&mut seed)?;let a=g.buffer(t*2048*2)?;let b=g.buffer(t*2048*2)?;
 run(&g,&q,&k,&v,&a,t,n,NAMES[0],1)?;
 for name in &NAMES[1..]{run(&g,&q,&k,&v,&b,t,n,name,1)?;let diffs=bytes(&a).chunks_exact(2).zip(bytes(&b).chunks_exact(2)).filter(|(x,y)|x!=y).count();failures+=diffs;println!("{{\"mode\":\"check\",\"rows\":{t},\"length\":{n},\"seed\":{si},\"variant\":\"{name}\",\"differences\":{diffs}}}");}
 if bench{for round in 0..4{let order:Vec<usize>=if round%2==0{(0..NAMES.len()).collect()}else{(0..NAMES.len()).rev().collect()};for i in order{let name=NAMES[i];let us=run(&g,&q,&k,&v,&b,t,n,name,2)?;println!("{{\"mode\":\"bench\",\"rows\":{t},\"length\":{n},\"seed\":{si},\"round\":{round},\"variant\":\"{name}\",\"us\":{us}}}");}}}
 }}if failures>0{bail!("{failures} BF16 differences");}Ok(())
}
