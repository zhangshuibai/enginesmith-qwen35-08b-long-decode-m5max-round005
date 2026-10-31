use crate::gpu::{Buf, Cmd, Enc, Gpu};
use crate::weights::{Linear, Weights};
use anyhow::{ensure, Result};
use std::{collections::VecDeque, path::Path, sync::{Arc, atomic::{AtomicBool, Ordering}}};
use objc2_metal::{MTLBuffer, MTLCommandEncoder};

pub const CHUNK: usize=2048;
pub const CAPACITY: usize=262144;
pub const VOCAB: usize=248320;

pub struct Gdn {
    qkv: Linear, z: Linear, a: Linear, b: Linear, out: Linear,
    conv: Buf, norm: Buf, dt: Buf, alog: Buf,
    history: Buf, state: Buf,
}
pub struct Attention {
    q: Linear, k: Linear, v: Linear, out: Linear,
    qnorm: Buf, knorm: Buf, keys: Buf, values: Buf,
}
pub enum Mixer { Gdn(Gdn), Attention(Attention) }
pub struct Layer { input_norm: Buf, post_norm: Buf, gate: Linear, up: Linear, down: Linear, mixer: Mixer }
pub struct Scratch {
    h: Buf, u: Buf, residual: Buf, qkv: Buf, z: Buf, a: Buf, b: Buf,
    conv: Buf, decay: Buf, beta: Buf, mixed: Buf, gated: Buf,
    q: Buf, k: Buf, v: Buf, query: Buf,
    gate: Buf, up: Buf, activated: Buf, logits: Buf, partial: Buf, sums: Buf, maxima: Buf,
}
pub struct Model {
    pub gpu: Gpu,
    embed: Linear, norm: Buf, layers: Vec<Layer>, scratch: Scratch,
    prompt_ids: Buf, output_ids: Buf,
    pub position: usize, pending: Option<u32>,
    pub cancelled: Arc<AtomicBool>,
}

impl Model {
    pub fn load(path: &Path) -> Result<Self> {
        let mut gpu=Gpu::new()?;
        compile_kernels(&mut gpu)?;
        let mut weights=Weights::load(&gpu,path)?;
        let embed=weights.linear("embed_tokens",VOCAB,1024)?;
        let norm=weights.take("norm.weight",&[1024],"BF16")?;
        let mut layers=Vec::new();
        for l in 0..24 {
            let p=format!("layers.{l}");
            let input_norm=weights.take(&format!("{p}.input_layernorm.weight"),&[1024],"BF16")?;
            let post_norm=weights.take(&format!("{p}.post_attention_layernorm.weight"),&[1024],"BF16")?;
            let gate=weights.linear(&format!("{p}.mlp.gate_proj"),3584,1024)?;
            let up=weights.linear(&format!("{p}.mlp.up_proj"),3584,1024)?;
            let down=weights.linear(&format!("{p}.mlp.down_proj"),1024,3584)?;
            let mixer=if l%4==3 {
                let p=format!("{p}.self_attn");
                Mixer::Attention(Attention {
                    q:weights.linear(&format!("{p}.q_proj"),4096,1024)?,
                    k:weights.linear(&format!("{p}.k_proj"),512,1024)?,
                    v:weights.linear(&format!("{p}.v_proj"),512,1024)?,
                    out:weights.linear(&format!("{p}.o_proj"),1024,2048)?,
                    qnorm:weights.take(&format!("{p}.q_norm.weight"),&[256],"BF16")?,
                    knorm:weights.take(&format!("{p}.k_norm.weight"),&[256],"BF16")?,
                    keys:gpu.buffer(2*CAPACITY*256*2)?,values:gpu.buffer(2*CAPACITY*256*2)?,
                })
            } else {
                let p=format!("{p}.linear_attn");
                Mixer::Gdn(Gdn {
                    qkv:weights.linear(&format!("{p}.in_proj_qkv"),6144,1024)?,
                    z:weights.linear(&format!("{p}.in_proj_z"),2048,1024)?,
                    a:weights.linear(&format!("{p}.in_proj_a"),16,1024)?,
                    b:weights.linear(&format!("{p}.in_proj_b"),16,1024)?,
                    out:weights.linear(&format!("{p}.out_proj"),1024,2048)?,
                    conv:weights.take(&format!("{p}.conv1d.weight"),&[6144,4,1],"BF16")?,
                    norm:weights.take(&format!("{p}.norm.weight"),&[128],"BF16")?,
                    dt:weights.take(&format!("{p}.dt_bias"),&[16],"BF16")?,
                    alog:weights.take(&format!("{p}.A_log"),&[16],"F32")?,
                    history:gpu.buffer(3*6144*2)?,state:gpu.buffer(16*128*128*4)?,
                })
            };
            layers.push(Layer{input_norm,post_norm,gate,up,down,mixer});
        }
        weights.done()?;
        let bf=|width|gpu.buffer(CHUNK*width*2);
        let scratch=Scratch {
            h:bf(1024)?,u:bf(1024)?,residual:bf(1024)?,qkv:bf(6144)?,z:bf(2048)?,a:bf(16)?,b:bf(16)?,
            conv:bf(6144)?,decay:gpu.buffer(CHUNK*16*4)?,beta:bf(16)?,mixed:bf(2048)?,gated:bf(2048)?,
            q:bf(4096)?,k:bf(512)?,v:bf(512)?,query:bf(2048)?,gate:bf(3584)?,up:bf(3584)?,activated:bf(3584)?,
            logits:gpu.buffer(VOCAB*2)?,partial:gpu.buffer(8*64*256*4)?,sums:gpu.buffer(8*64*4)?,maxima:gpu.buffer(8*64*4)?,
        };
        let prompt_ids=gpu.buffer(CAPACITY*4)?;
        let output_ids=gpu.buffer(CAPACITY*4)?;
        let mut m=Self{gpu,embed,norm,layers,scratch,prompt_ids,output_ids,position:0,pending:None,cancelled:Arc::new(AtomicBool::new(false))};
        m.reset();
        Ok(m)
    }
    fn reset(&mut self) {
        self.position=0;self.pending=None;
        for l in &self.layers {if let Mixer::Gdn(g)=&l.mixer {Gpu::zero(&g.history);Gpu::zero(&g.state);}}
    }
    pub fn generate(&mut self,prompt:&[u32],count:usize,reset:bool)->Result<Vec<u32>> {
        ensure!(count>0&&count<=CAPACITY,"invalid output length");
        let first_offset;
        if reset {
            ensure!(!prompt.is_empty(),"reset needs nonempty prompt");
            ensure!(prompt.len()+count-1<=CAPACITY,"context capacity exceeded");
            ensure!(prompt.iter().all(|&x|x<(VOCAB as u32)),"token outside vocabulary");
            self.reset();
            unsafe {std::ptr::copy_nonoverlapping(prompt.as_ptr(),self.prompt_ids.contents().as_ptr().cast::<u32>(),prompt.len());}
            while self.position+1<prompt.len() {
                let n=CHUNK.min(prompt.len()-self.position-1);
                self.forward(&self.prompt_ids.clone(),self.position*4,n,None)?;
            }
            first_offset=self.position*4;
        } else {
            ensure!(prompt.is_empty(),"continuation prompt must be empty");
            ensure!(self.position+count<=CAPACITY,"context capacity exceeded");
            let pending=self.pending.ok_or_else(||anyhow::anyhow!("no continuation state"))?;
            unsafe {*self.prompt_ids.contents().as_ptr().cast::<u32>()=pending;}
            first_offset=0;
        }
        self.decode(count,first_offset)?;
        let out=unsafe{std::slice::from_raw_parts(self.output_ids.contents().as_ptr().cast::<u32>(),count).to_vec()};
        self.pending=out.last().copied();
        Ok(out)
    }
    fn decode(&mut self,count:usize,first_offset:usize)->Result<()> {
        // The token tail already writes output_ids on the GPU. Encode its
        // consumers ahead on the same queue; all scratch and state are ordered
        // by the serial encoders. Keep the outstanding work bounded on EOF.
        const AHEAD: usize=16;
        let mut inflight=VecDeque::<(Cmd,usize)>::new();
        let mut outstanding=0;
        let mut next=0;
        let encoded=(||->Result<()> {
            while next<count {
                ensure!(!self.cancelled.load(Ordering::Relaxed),"input disconnected");
                let batch=if next==0 {1} else {4.min(count-next)};
                while outstanding+batch>AHEAD {
                    let (cmd,steps)=inflight.pop_front().expect("outstanding decode work");
                    outstanding-=steps;
                    Gpu::wait(&cmd)?;
                }
                ensure!(!self.cancelled.load(Ordering::Relaxed),"input disconnected");
                let (cmd,enc)=self.gpu.begin()?;
                let start=next;
                for i in start..start+batch {
                    if self.cancelled.load(Ordering::Relaxed) {break;}
                    if i==0 {
                        self.encode_forward(&enc,&self.prompt_ids,first_offset,1,Some(i));
                    } else {
                        self.encode_forward(&enc,&self.output_ids,(i-1)*4,1,Some(i));
                    }
                    // Position is an encode-time parameter; GPU state reaches
                    // it only after every submitted command buffer is drained.
                    self.position+=1;
                    next+=1;
                }
                let steps=next-start;
                if steps==0 {
                    enc.endEncoding();
                } else {
                    inflight.push_back((Gpu::commit(cmd,enc),steps));
                    outstanding+=steps;
                }
                ensure!(!self.cancelled.load(Ordering::Relaxed),"input disconnected");
            }
            Ok(())
        })();
        // Never return with queued writes to reusable state. Even if an early
        // command fails, wait and inspect every later committed command.
        let mut error=encoded.err();
        for (cmd,_) in inflight {
            if let Err(e)=Gpu::wait(&cmd) {if error.is_none() {error=Some(e);}}
        }
        if error.is_none()&&self.cancelled.load(Ordering::Relaxed) {
            error=Some(anyhow::anyhow!("input disconnected"));
        }
        if let Some(e)=error {
            self.reset();
            return Err(e);
        }
        Ok(())
    }
    fn forward(&mut self,ids:&Buf,id_offset:usize,rows:usize,output:Option<usize>)->Result<()> {
        ensure!(!self.cancelled.load(Ordering::Relaxed),"input disconnected");
        let (cmd,enc)=self.gpu.begin()?;
        self.encode_forward(&enc,ids,id_offset,rows,output);
        Gpu::finish(cmd,enc)?;
        self.position+=rows;
        Ok(())
    }
    fn linear(&self,enc:&Enc,w:&Linear,x:&Buf,y:&Buf,rows:usize) {
        // MLX 0.32.4 get_qmv_batch_limit for this family-10 applegpu_g17s.
        let vector_limit=if w.k<=2048 && w.n<=2048 {33}
            else if w.k<=4096 && w.n<=4096 {25} else {13};
        let (kernel,groups,threads)=if rows<vector_limit {
            ("qmv_q4",[rows,w.n.div_ceil(8),1],[64,1,1])
        } else {("qmm_q4_nax",[w.n.div_ceil(64),rows.div_ceil(64),1],[128,1,1])};
        self.gpu.dispatch(enc,kernel,&[(&w.w,0),(&w.s,0),(&w.b,0),(x,0),(y,0)],
            &[rows as u32,w.n as u32,w.k as u32],groups,threads);
    }
    // Decode input normalization repeats the exact eight-SIMD RMS in each
    // group; projections keep their original four-row SIMD reduction. The
    // raw residual, normalized scratch and projection outputs are distinct.
    fn linear_group_norm(&self,enc:&Enc,w:[&Linear;4],x:&Buf,gain:&Buf,norm_out:&Buf,y:[&Buf;4],count:usize) {
        let n3=if count==4 {w[3].n} else {0};
        let groups=w[..count].iter().map(|m|m.n.div_ceil(32)).sum();
        self.gpu.dispatch(enc,"qmv_multi_g8_norm",
            &[(&w[0].w,0),(&w[0].s,0),(&w[0].b,0),(y[0],0),
              (&w[1].w,0),(&w[1].s,0),(&w[1].b,0),(y[1],0),
              (&w[2].w,0),(&w[2].s,0),(&w[2].b,0),(y[2],0),
              (&w[3].w,0),(&w[3].s,0),(&w[3].b,0),(y[3],0),(x,0),(gain,0),(norm_out,0)],
            &[1,w[0].n as u32,w[1].n as u32,w[2].n as u32,n3 as u32,w[0].k as u32],
            [1,groups,1],[256,1,1]);
    }
    fn linear_add(&self,enc:&Enc,w:&Linear,x:&Buf,residual:&Buf) {
        // Every QMV row owns its residual element. x is a distinct buffer,
        // so residual may safely be both an input and the output here.
        self.gpu.dispatch(enc,"qmv_add_q4",
            &[(&w.w,0),(&w.s,0),(&w.b,0),(x,0),(residual,0),(residual,0)],
            &[1,w.n as u32,w.k as u32],[1,w.n.div_ceil(8),1],[64,1,1]);
    }
    fn norm(&self,enc:&Enc,x:&Buf,w:&Buf,y:&Buf,rows:usize) {
        self.gpu.dispatch(enc,"rms_norm_bf16",&[(x,0),(w,0),(y,0)],
            &[rows as u32,1024,1e-6f32.to_bits()],[rows,1,1],[256,1,1]);
    }
    fn add(&self,enc:&Enc,a:&Buf,b:&Buf,out:&Buf,count:usize) {
        self.gpu.dispatch(enc,"add_bf16",&[(a,0),(b,0),(out,0)],&[count as u32],
            [count.div_ceil(256),1,1],[256,1,1]);
    }
    fn encode_forward(&self,enc:&Enc,ids:&Buf,id_offset:usize,rows:usize,output:Option<usize>) {
        let s=&self.scratch;
        let e=&self.embed;
        self.gpu.dispatch(enc,"embedding_q4",&[(&e.w,0),(&e.s,0),(&e.b,0),(ids,id_offset),(&s.h,0)],
            &[rows as u32,VOCAB as u32,1024],[(rows*1024).div_ceil(256),1,1],[256,1,1]);
        for layer in &self.layers {
            if rows!=1 {self.norm(enc,&s.h,&layer.input_norm,&s.u,rows);}
            match &layer.mixer {
                Mixer::Gdn(g)=>{
                    if rows==1 {
                        // Each group owns its complete 128-channel convolution
                        // block, so history and gates have no grid-wide dependency.
                        self.gpu.dispatch(enc,"gdn_input_g16",
                            &[(&g.qkv.w,0),(&g.qkv.s,0),(&g.qkv.b,0),(&s.qkv,0),
                              (&g.z.w,0),(&g.z.s,0),(&g.z.b,0),(&s.z,0),
                              (&g.a.w,0),(&g.a.s,0),(&g.a.b,0),(&s.a,0),
                              (&g.b.w,0),(&g.b.s,0),(&g.b.b,0),(&s.b,0),
                              (&s.h,0),(&layer.input_norm,0),(&s.u,0),(&g.history,0),
                              (&g.conv,0),(&s.conv,0),(&g.alog,0),(&g.dt,0),
                              (&s.decay,0),(&s.beta,0)],
                            &[1,6144,2048,16,16,1024],[1,64,1],[512,1,1]);
                    } else {
                        self.linear(enc,&g.qkv,&s.u,&s.qkv,rows);
                        self.linear(enc,&g.z,&s.u,&s.z,rows);
                        self.linear(enc,&g.a,&s.u,&s.a,rows);
                        self.linear(enc,&g.b,&s.u,&s.b,rows);
                        self.gpu.dispatch(enc,"gdn_conv",&[(&s.qkv,0),(&g.history,0),(&g.conv,0),(&s.conv,0)],
                            &[rows as u32],[(rows*6144).div_ceil(256),1,1],[256,1,1]);
                        self.gpu.dispatch(enc,"gdn_history",&[(&s.qkv,0),(&g.history,0)],
                            &[rows as u32],[24,1,1],[256,1,1]);
                        self.gpu.dispatch(enc,"gdn_prepare",&[(&s.conv,0),(&s.a,0),(&s.b,0),(&g.alog,0),(&g.dt,0),(&s.decay,0),(&s.beta,0)],
                            &[rows as u32],[rows*16,1,1],[32,1,1]);
                    }
                    self.gpu.dispatch(enc,"gdn_recur",&[(&s.conv,0),(&s.decay,0),(&s.beta,0),(&g.state,0),(&s.mixed,0)],
                        &[rows as u32],[1,8,16],[32,2,1]);
                    self.gpu.dispatch(enc,"gdn_norm_gate",&[(&s.mixed,0),(&s.z,0),(&g.norm,0),(&s.gated,0)],
                        &[rows as u32],[rows*16,1,1],[32,1,1]);
                    if rows==1 {self.linear_add(enc,&g.out,&s.gated,&s.h);}
                    else {self.linear(enc,&g.out,&s.gated,&s.residual,rows);}
                }
                Mixer::Attention(a)=>self.attention(enc,a,rows,&layer.input_norm),
            }
            if rows!=1 {self.add(enc,&s.h,&s.residual,&s.h,rows*1024);}
            if rows!=1 {self.norm(enc,&s.h,&layer.post_norm,&s.u,rows);}
            if rows==1 {
                self.gpu.dispatch(enc,"qmv_mlp_g8_norm",
                    &[(&layer.gate.w,0),(&layer.gate.s,0),(&layer.gate.b,0),
                      (&layer.up.w,0),(&layer.up.s,0),(&layer.up.b,0),(&s.h,0),(&s.activated,0),
                      (&layer.post_norm,0),(&s.u,0)],
                    &[1,3584,1024],[1,224,1],[256,1,1]);
            } else {
                self.linear(enc,&layer.gate,&s.u,&s.gate,rows);
                self.linear(enc,&layer.up,&s.u,&s.up,rows);
                self.gpu.dispatch(enc,"swiglu_bf16",&[(&s.gate,0),(&s.up,0),(&s.activated,0)],
                    &[(rows*3584) as u32],[(rows*3584).div_ceil(256),1,1],[256,1,1]);
            }
            if rows==1 {self.linear_add(enc,&layer.down,&s.activated,&s.h);}
            else {
                self.linear(enc,&layer.down,&s.activated,&s.residual,rows);
                self.add(enc,&s.h,&s.residual,&s.h,rows*1024);
            }
        }
        if let Some(i)=output {
            self.norm(enc,&s.h,&self.norm,&s.u,rows);
            if rows==1 {
                // The attention temporary is dead at the head. Two disjoint 1024-F32
                // slices retain the original greedy threads' online LSE partials.
                self.gpu.dispatch(enc,"head_lse_g8",&[(&e.w,0),(&e.s,0),(&e.b,0),
                    (&s.u,0),(&s.logits,0),(&s.partial,0),(&s.partial,4096)],
                    &[1,VOCAB as u32,1024],[128,1,1],[256,1,1]);
                self.gpu.dispatch(enc,"head_lse_tail",&[(&s.logits,0),(&s.partial,0),
                    (&s.partial,4096),(&self.output_ids,i*4)],
                    &[VOCAB as u32],[1,1,1],[1024,1,1]);
            } else {
                self.linear(enc,&self.embed,&s.u,&s.logits,rows);
                self.gpu.dispatch(enc,"greedy_bf16",&[(&s.logits,0),(&self.output_ids,i*4)],
                    &[VOCAB as u32],[1,1,1],[1024,1,1]);
            }
        }
    }
    fn attention(&self,enc:&Enc,a:&Attention,rows:usize,input_norm:&Buf) {
        let s=&self.scratch;
        let p=[rows as u32,self.position as u32,CAPACITY as u32];
        if rows==1 {
            self.linear_group_norm(enc,[&a.q,&a.k,&a.v,&a.v],&s.h,input_norm,&s.u,[&s.q,&s.k,&s.v,&s.v],3);
        } else {
            self.linear(enc,&a.q,&s.u,&s.q,rows);
            self.linear(enc,&a.k,&s.u,&s.k,rows);
            self.linear(enc,&a.v,&s.u,&s.v,rows);
        }
        if rows==1 {
            // Q and KV groups have disjoint outputs and no mutual dependency.
            self.gpu.dispatch(enc,"attn_prepare_qkv_decode",
                &[(&s.q,0),(&s.k,0),(&s.v,0),(&a.qnorm,0),(&a.knorm,0),
                  (&s.query,0),(&a.keys,0),(&a.values,0)],&p,[10,1,1],[64,1,1]);
        } else {
            self.gpu.dispatch(enc,"attn_prepare_q",&[(&s.q,0),(&a.qnorm,0),(&s.query,0)],&p,[8,rows,1],[64,1,1]);
            self.gpu.dispatch(enc,"attn_prepare_kv",&[(&s.k,0),(&s.v,0),(&a.knorm,0),(&a.keys,0),(&a.values,0)],&p,[2,rows,1],[64,1,1]);
        }
        let len=self.position+rows;
        if rows==1 {
            let p=[len as u32,CAPACITY as u32];
            if len<1024 {
                self.gpu.dispatch(enc,"attn_decode",&[(&s.query,0),(&a.keys,0),(&a.values,0),(&s.mixed,0)],&p,[8,1,1],[1024,1,1]);
            } else {
                self.gpu.dispatch(enc,"attn_decode_split1",&[(&s.query,0),(&a.keys,0),(&a.values,0),(&s.partial,0),(&s.sums,0),(&s.maxima,0)],&p,[2,1,64],[32,4,1]);
                self.gpu.dispatch(enc,"attn_decode_split2_gate",&[(&s.partial,0),(&s.sums,0),(&s.maxima,0),(&s.gated,0),(&s.q,0)],&p,[8,32,1],[32,1,1]);
            }
        } else {
            let mut p=vec![1,8,256,rows as u32,len as u32,4,(1.0f32/16.0).to_bits(),
                rows.div_ceil(64) as u32,len.div_ceil(32) as u32,(rows/64) as u32,(len/32) as u32,
                (rows%64) as u32,(len%32) as u32,self.position as u32];
            for stride in [rows*2048,256,2048,2*CAPACITY*256,CAPACITY*256,256,2*CAPACITY*256,CAPACITY*256,256,rows*2048,256,2048] {
                p.push(stride as u32);p.push((stride as u64>>32) as u32);
            }
            self.gpu.dispatch(enc,"attn_prefill",&[(&s.query,0),(&a.keys,0),(&a.values,0),(&s.mixed,0)],&p,[rows.div_ceil(64),8,1],[256,1,1]);
        }
        if rows!=1 || len<1024 {
            self.gpu.dispatch(enc,"attn_gate",&[(&s.mixed,0),(&s.q,0),(&s.gated,0)],&[rows as u32],[(rows*2048).div_ceil(256),1,1],[256,1,1]);
        }
        if rows==1 {self.linear_add(enc,&a.out,&s.gated,&s.h);}
        else {self.linear(enc,&a.out,&s.gated,&s.residual,rows);}
    }
}

fn compile_kernels(gpu: &mut Gpu)->Result<()> {
    gpu.compile(include_str!("kernels/head_lse.metal"),&["head_lse_g8","head_lse_tail"])?;
    gpu.compile(include_str!("kernels/linear_input_norm.metal"),&["qmv_multi_g8_norm"])?;
    gpu.compile(include_str!("kernels/linear_norm.metal"),&["qmv_mlp_g8_norm"])?;
    gpu.compile(include_str!("kernels/linear.metal"),&["embedding_q4","qmv_q4","qmv_add_q4","qmv_multi_q4","qmv_gate_up_swiglu_q4","rms_norm_bf16","add_bf16","swiglu_bf16","greedy_bf16"])?;
    gpu.compile(include_str!("kernels/linear_nax.metal"),&["qmm_q4_nax"])?;
    gpu.compile(include_str!("kernels/gdn.metal"),&["gdn_conv","gdn_history","gdn_prepare","gdn_conv_prepare_decode","gdn_recur","gdn_norm_gate"])?;
    gpu.compile(include_str!("kernels/gdn_input.metal"),&["gdn_input_g16"])?;
    gpu.compile(include_str!("kernels/attention.metal"),&["attn_prepare_q","attn_prepare_kv","attn_prepare_qkv_decode","attn_gate"])?;
    gpu.compile(include_str!("kernels/attention_decode.metal"),&["attn_decode","attn_decode_split1","attn_decode_split2_gate"])?;
    gpu.compile(include_str!("kernels/attention_prefill.metal"),&["attn_prefill"])?;
    Ok(())
}
