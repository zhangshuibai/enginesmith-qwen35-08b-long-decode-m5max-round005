use crate::gpu::{Buf, Gpu};
use anyhow::{anyhow, bail, ensure, Context, Result};
use serde_json::Value;
use std::{collections::HashMap, fs, path::Path};

pub struct Tensor { pub buf: Buf, pub shape: Vec<usize>, pub dtype: String }
pub struct Weights { tensors: HashMap<String, Tensor> }
pub struct Linear { pub w: Buf, pub s: Buf, pub b: Buf, pub n: usize, pub k: usize }

impl Weights {
    pub fn load(gpu: &Gpu, model: &Path) -> Result<Self> {
        let config: Value = serde_json::from_slice(&fs::read(model.join("config.json"))?)?;
        let tc = &config["text_config"];
        ensure!(tc["hidden_size"] == 1024 && tc["num_hidden_layers"] == 24 && tc["vocab_size"] == 248320,
            "unsupported model configuration");
        let mut tensors = HashMap::new();
        let mut files = Vec::new();
        for e in fs::read_dir(model)? { let p=e?.path(); if p.extension().is_some_and(|x|x=="safetensors") { files.push(p); } }
        files.sort();
        ensure!(!files.is_empty(), "no safetensors weights");
        for path in files {
            let bytes=fs::read(&path).with_context(|| format!("read {}",path.display()))?;
            ensure!(bytes.len()>=8,"short safetensors file");
            let n=u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
            ensure!(n<=bytes.len()-8,"invalid safetensors header");
            let header: Value=serde_json::from_slice(&bytes[8..8+n])?;
            for (name,v) in header.as_object().ok_or_else(||anyhow!("invalid header"))? {
                if name=="__metadata__" {continue;}
                if !name.starts_with("language_model.model.") {continue;}
                let name=name.strip_prefix("language_model.model.").unwrap().to_owned();
                let dtype=v["dtype"].as_str().ok_or_else(||anyhow!("missing dtype"))?.to_owned();
                let elem=match dtype.as_str(){"U32"|"F32"=>4,"BF16"=>2,_=>bail!("unsupported dtype {dtype}")};
                let shape:Vec<usize>=v["shape"].as_array().ok_or_else(||anyhow!("missing shape"))?.iter()
                    .map(|x|x.as_u64().map(|x|x as usize).ok_or_else(||anyhow!("invalid shape"))).collect::<Result<_>>()?;
                let start=v["data_offsets"][0].as_u64().ok_or_else(||anyhow!("invalid offset"))? as usize;
                let end=v["data_offsets"][1].as_u64().ok_or_else(||anyhow!("invalid offset"))? as usize;
                ensure!(end>=start&&end<=bytes.len()-8-n&&end-start==shape.iter().product::<usize>()*elem,"tensor bounds: {name}");
                let buf=gpu.from_bytes(&bytes[8+n+start..8+n+end])?;
                ensure!(tensors.insert(name.clone(),Tensor{buf,shape,dtype}).is_none(),"duplicate tensor {name}");
            }
        }
        ensure!(tensors.len()==694,"expected 694 text tensors, got {}",tensors.len());
        Ok(Self{tensors})
    }
    pub fn take(&mut self,name:&str,shape:&[usize],dtype:&str)->Result<Buf>{
        let t=self.tensors.remove(name).ok_or_else(||anyhow!("missing {name}"))?;
        ensure!(t.shape==shape&&t.dtype==dtype,"wrong shape/dtype for {name}");
        Ok(t.buf)
    }
    pub fn linear(&mut self,name:&str,n:usize,k:usize)->Result<Linear>{
        Ok(Linear{w:self.take(&format!("{name}.weight"),&[n,k/8],"U32")?,s:self.take(&format!("{name}.scales"),&[n,k/64],"BF16")?,b:self.take(&format!("{name}.biases"),&[n,k/64],"BF16")?,n,k})
    }
    pub fn done(self)->Result<()> { ensure!(self.tensors.is_empty(),"unused text weights: {:?}",self.tensors.keys());Ok(()) }
}
