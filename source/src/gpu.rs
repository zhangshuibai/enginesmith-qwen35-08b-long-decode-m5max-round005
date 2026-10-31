use anyhow::{anyhow, bail, Result};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::NSString;
use objc2_metal::*;
use std::{collections::HashMap, ffi::c_void, ptr::NonNull};

pub type Buf = Retained<ProtocolObject<dyn MTLBuffer>>;
pub type Enc = ProtocolObject<dyn MTLComputeCommandEncoder>;
pub type Cmd = Retained<ProtocolObject<dyn MTLCommandBuffer>>;

pub struct Gpu {
    pub device: Retained<ProtocolObject<dyn MTLDevice>>,
    pub queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pipelines: HashMap<String, Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
}

impl Gpu {
    pub fn new() -> Result<Self> {
        let device = MTLCreateSystemDefaultDevice().ok_or_else(|| anyhow!("Metal device unavailable"))?;
        if !device.supportsFamily(MTLGPUFamily::Apple10) { bail!("requires Apple GPU family 10"); }
        let queue = device.newCommandQueue().ok_or_else(|| anyhow!("Metal queue unavailable"))?;
        Ok(Self { device, queue, pipelines: HashMap::new() })
    }
    pub fn compile(&mut self, source: &str, names: &[&str]) -> Result<()> {
        let opts = MTLCompileOptions::new();
        opts.setLanguageVersion(MTLLanguageVersion::Version4_0);
        #[allow(deprecated)]
        opts.setFastMathEnabled(false);
        let lib = self.device.newLibraryWithSource_options_error(&NSString::from_str(source), Some(&opts))
            .map_err(|e| anyhow!("Metal compile: {e}"))?;
        for &name in names {
            let f = lib.newFunctionWithName(&NSString::from_str(name)).ok_or_else(|| anyhow!("missing kernel {name}"))?;
            let p = self.device.newComputePipelineStateWithFunction_error(&f).map_err(|e| anyhow!("pipeline {name}: {e}"))?;
            if p.threadExecutionWidth() != 32 { bail!("unsupported SIMD width"); }
            self.pipelines.insert(name.into(), p);
        }
        Ok(())
    }
    pub fn buffer(&self, bytes: usize) -> Result<Buf> {
        self.device.newBufferWithLength_options(bytes.max(4), MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| anyhow!("Metal buffer allocation of {bytes} bytes failed"))
    }
    pub fn from_bytes(&self, bytes: &[u8]) -> Result<Buf> {
        let b = self.buffer(bytes.len())?;
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), b.contents().as_ptr().cast(), bytes.len()); }
        Ok(b)
    }
    pub fn zero(b: &Buf) {
        unsafe { std::ptr::write_bytes(b.contents().as_ptr().cast::<u8>(), 0, b.length()); }
    }
    pub fn begin(&self) -> Result<(Cmd, Retained<Enc>)> {
        let cmd = self.queue.commandBuffer().ok_or_else(|| anyhow!("command buffer unavailable"))?;
        let enc = cmd.computeCommandEncoder().ok_or_else(|| anyhow!("encoder unavailable"))?;
        Ok((cmd, enc))
    }
    pub fn commit(cmd: Cmd, enc: Retained<Enc>) -> Cmd {
        enc.endEncoding();
        cmd.commit();
        cmd
    }
    pub fn wait(cmd: &Cmd) -> Result<()> {
        cmd.waitUntilCompleted();
        if cmd.status() == MTLCommandBufferStatus::Error { bail!("GPU command failed: {:?}", cmd.error()); }
        Ok(())
    }
    pub fn finish(cmd: Cmd, enc: Retained<Enc>) -> Result<()> {
        Self::wait(&Self::commit(cmd, enc))
    }
    pub fn dispatch(&self, enc: &Enc, name: &str, buffers: &[(&Buf, usize)], params: &[u32], groups: [usize; 3], threads: [usize; 3]) {
        enc.setComputePipelineState(&self.pipelines[name]);
        for (i, (b, offset)) in buffers.iter().enumerate() {
            assert!(offset % 4 == 0);
            unsafe { enc.setBuffer_offset_atIndex(Some(b), *offset, i); }
        }
        if !params.is_empty() {
            unsafe { enc.setBytes_length_atIndex(NonNull::new(params.as_ptr() as *mut c_void).unwrap(), params.len()*4, buffers.len()); }
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(size(groups), size(threads));
    }
}

fn size(x: [usize; 3]) -> MTLSize { MTLSize { width: x[0], height: x[1], depth: x[2] } }
