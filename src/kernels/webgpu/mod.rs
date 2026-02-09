//! WebGPU Backend
//!
//! This module provides GPU-accelerated inference using the WebGPU API via the wgpu crate.
//! It implements the `KernelBackend` trait for seamless integration with the rest of the system.

pub mod memory;
pub mod shaders;
pub mod tensor;

pub use memory::WebGPUMemoryPool;
pub use tensor::WebGPUTensor;

use crate::kernels::backend::KernelBackend;
use crate::tensor::{Tensor1, Tensor2};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut1};
use std::sync::Arc;
use wgpu::util::DeviceExt;

// =============================================================================
// WebGPU Backend Implementation
// =============================================================================

/// WebGPU backend for cross-platform GPU inference.
#[derive(Clone)]
pub struct WebGPUBackend {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    pipelines: Arc<WebGPUPipelines>,
    #[allow(dead_code)]
    memory_pool: Arc<std::sync::Mutex<WebGPUMemoryPool>>,
    device_name: String,
}

/// Container for WebGPU compute pipelines.
struct WebGPUPipelines {
    rmsnorm: wgpu::ComputePipeline,
    softmax: wgpu::ComputePipeline,
    silu: wgpu::ComputePipeline,
    rope: wgpu::ComputePipeline,
    attention_scores: wgpu::ComputePipeline,
    weighted_sum: wgpu::ComputePipeline,
    matmul_vec: wgpu::ComputePipeline,
}

impl std::fmt::Debug for WebGPUBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WebGPUBackend")
            .field("device_name", &self.device_name)
            .finish()
    }
}

impl WebGPUBackend {
    /// Check if WebGPU is available on this system.
    pub fn is_available() -> bool {
        pollster::block_on(async {
            let instance = wgpu::Instance::default();
            instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                })
                .await
                .is_ok()
        })
    }

    /// Create a new WebGPU backend.
    pub fn new() -> anyhow::Result<Self> {
        let (device, queue, device_name) = pollster::block_on(Self::request_device())?;

        let device = Arc::new(device);
        let queue = Arc::new(queue);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("torchless WebGPU shaders"),
            source: wgpu::ShaderSource::Wgsl(shaders::WGSL_SHADERS_SOURCE.into()),
        });

        let pipelines = Self::create_pipelines(&device, &shader)?;

        Ok(Self {
            device: device.clone(),
            queue: queue.clone(),
            pipelines: Arc::new(pipelines),
            memory_pool: Arc::new(std::sync::Mutex::new(WebGPUMemoryPool::new())),
            device_name,
        })
    }

    async fn request_device() -> anyhow::Result<(wgpu::Device, wgpu::Queue, String)> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .map_err(|e| anyhow::anyhow!("No WebGPU adapter available: {:?}", e))?;

        let device_name = adapter.get_info().name.clone();

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("torchless WebGPU device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: Default::default(),
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(|e| anyhow::anyhow!("Device creation failed: {:?}", e))?;

        Ok((device, queue, device_name))
    }

    fn create_pipelines(
        device: &wgpu::Device,
        shader: &wgpu::ShaderModule,
    ) -> anyhow::Result<WebGPUPipelines> {
        let create_pipeline = |entry_point: &str| -> anyhow::Result<wgpu::ComputePipeline> {
            Ok(
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(entry_point),
                    layout: None,
                    module: shader,
                    entry_point: Some(entry_point),
                    compilation_options: Default::default(),
                    cache: None,
                }),
            )
        };

        Ok(WebGPUPipelines {
            rmsnorm: create_pipeline(shaders::kernel_names::RMSNORM)?,
            softmax: create_pipeline(shaders::kernel_names::SOFTMAX)?,
            silu: create_pipeline(shaders::kernel_names::SILU)?,
            rope: create_pipeline(shaders::kernel_names::ROPE)?,
            attention_scores: create_pipeline(shaders::kernel_names::ATTENTION_SCORES)?,
            weighted_sum: create_pipeline(shaders::kernel_names::WEIGHTED_SUM)?,
            matmul_vec: create_pipeline(shaders::kernel_names::MATMUL_VEC)?,
        })
    }

    /// Get the name of the GPU device.
    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    /// Create a buffer from CPU data.
    pub fn create_buffer(&self, data: &[f32]) -> wgpu::Buffer {
        memory::create_buffer_with_data(&self.device, data)
    }

    /// Create a zero-initialized buffer.
    pub fn create_buffer_zeros(&self, len: usize) -> wgpu::Buffer {
        memory::create_buffer_zeros(&self.device, len)
    }

    /// Read data from a GPU buffer back to CPU.
    pub fn read_buffer(&self, buffer: &wgpu::Buffer, len: usize) -> Vec<f32> {
        memory::read_buffer_to_vec(&self.device, &self.queue, buffer, len)
    }

    // =========================================================================
    // GPU Kernel Launches
    // =========================================================================

    fn launch_rmsnorm(&self, x: &wgpu::Buffer, weight: &wgpu::Buffer, n: usize, eps: f32) {
        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            n: u32,
            eps: f32,
        }
        let params = Params { n: n as u32, eps };

        let params_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("rmsnorm params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let layout = self.pipelines.rmsnorm.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rmsnorm bind group"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: x.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: weight.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("rmsnorm encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("rmsnorm pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.rmsnorm);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        self.device.poll(wgpu::PollType::wait_indefinitely()).ok();
    }

    fn launch_softmax(&self, x: &wgpu::Buffer, n: usize) {
        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            n: u32,
        }
        let params = Params { n: n as u32 };

        let params_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("softmax params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let layout = self.pipelines.softmax.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("softmax bind group"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: x.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("softmax encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("softmax pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.softmax);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        self.device.poll(wgpu::PollType::wait_indefinitely()).ok();
    }

    fn launch_silu(&self, x: &wgpu::Buffer, y: &wgpu::Buffer, n: usize) {
        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            n: u32,
        }
        let params = Params { n: n as u32 };

        let params_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("silu params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let layout = self.pipelines.silu.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("silu bind group"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: x.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: y.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let workgroups = n.div_ceil(256);

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("silu encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("silu pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.silu);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroups as u32, 1, 1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        self.device.poll(wgpu::PollType::wait_indefinitely()).ok();
    }

    fn launch_rope(
        &self,
        x: &wgpu::Buffer,
        cos: &wgpu::Buffer,
        sin: &wgpu::Buffer,
        n_heads: usize,
        head_dim: usize,
    ) {
        let half = head_dim / 2;
        let total = n_heads * half;

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            n_heads: u32,
            head_dim: u32,
            half_dim: u32,
        }
        let params = Params {
            n_heads: n_heads as u32,
            head_dim: head_dim as u32,
            half_dim: half as u32,
        };

        let params_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("rope params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let layout = self.pipelines.rope.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rope bind group"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: x.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: cos.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: sin.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let workgroups = total.div_ceil(256);

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("rope encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("rope pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.rope);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroups as u32, 1, 1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        self.device.poll(wgpu::PollType::wait_indefinitely()).ok();
    }

    fn launch_attention_scores(
        &self,
        query: &wgpu::Buffer,
        keys: &wgpu::Buffer,
        scores: &wgpu::Buffer,
        seq_len: usize,
        head_dim: usize,
        scale: f32,
    ) {
        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            seq_len: u32,
            head_dim: u32,
            scale: f32,
        }
        let params = Params {
            seq_len: seq_len as u32,
            head_dim: head_dim as u32,
            scale,
        };

        let params_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("attention_scores params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let layout = self.pipelines.attention_scores.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("attention_scores bind group"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: query.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: keys.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: scores.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let workgroups = seq_len.div_ceil(256);

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("attention_scores encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("attention_scores pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.attention_scores);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroups as u32, 1, 1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        self.device.poll(wgpu::PollType::wait_indefinitely()).ok();
    }

    fn launch_weighted_sum(
        &self,
        weights: &wgpu::Buffer,
        matrix: &wgpu::Buffer,
        out: &wgpu::Buffer,
        n: usize,
        d: usize,
    ) {
        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            n: u32,
            d: u32,
        }
        let params = Params {
            n: n as u32,
            d: d as u32,
        };

        let params_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("weighted_sum params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let layout = self.pipelines.weighted_sum.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("weighted_sum bind group"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: weights.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: matrix.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let workgroups = d.div_ceil(256);

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("weighted_sum encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("weighted_sum pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.weighted_sum);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroups as u32, 1, 1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        self.device.poll(wgpu::PollType::wait_indefinitely()).ok();
    }

    fn launch_matmul_vec(
        &self,
        weights: &wgpu::Buffer,
        x: &wgpu::Buffer,
        out: &wgpu::Buffer,
        rows: usize,
        cols: usize,
    ) {
        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            rows: u32,
            cols: u32,
        }
        let params = Params {
            rows: rows as u32,
            cols: cols as u32,
        };

        let params_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("matmul_vec params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let layout = self.pipelines.matmul_vec.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("matmul_vec bind group"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: weights.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: x.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let workgroups = rows.div_ceil(256);

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("matmul_vec encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("matmul_vec pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.matmul_vec);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroups as u32, 1, 1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        self.device.poll(wgpu::PollType::wait_indefinitely()).ok();
    }
}

// =============================================================================
// KernelBackend Implementation
// =============================================================================

impl KernelBackend for WebGPUBackend {
    type Tensor1 = Tensor1;
    type Tensor2 = Tensor2;

    fn name(&self) -> &'static str {
        "webgpu"
    }

    fn is_available() -> bool {
        WebGPUBackend::is_available()
    }

    fn matmul_vec(&self, w: &Array2<f32>, x: &Array1<f32>) -> Array1<f32> {
        let (rows, cols) = w.dim();

        let w_buffer = self.create_buffer(w.as_slice().expect("Weight matrix must be contiguous"));
        let x_buffer = self.create_buffer(x.as_slice().expect("Input vector must be contiguous"));
        let out_buffer = self.create_buffer_zeros(rows);

        self.launch_matmul_vec(&w_buffer, &x_buffer, &out_buffer, rows, cols);

        Array1::from_vec(self.read_buffer(&out_buffer, rows))
    }

    fn matmul_vec_into(&self, w: &Array2<f32>, x: &Array1<f32>, out: &mut Array1<f32>) {
        let result = self.matmul_vec(w, x);
        out.assign(&result);
    }

    fn matmul(&self, a: &Array2<f32>, b: &Array2<f32>) -> Array2<f32> {
        let (_m, k1) = a.dim();
        let (k2, _n) = b.dim();
        assert_eq!(k1, k2, "Matrix dimensions must match for multiplication");
        a.dot(b)
    }

    fn rmsnorm(&self, x: &mut Array1<f32>, weight: &Array1<f32>, eps: f32) {
        let n = x.len();

        let x_buffer = self.create_buffer(x.as_slice().expect("x must be contiguous"));
        let weight_buffer =
            self.create_buffer(weight.as_slice().expect("weight must be contiguous"));

        self.launch_rmsnorm(&x_buffer, &weight_buffer, n, eps);

        let result = self.read_buffer(&x_buffer, n);
        x.as_slice_mut()
            .expect("x must be contiguous")
            .copy_from_slice(&result);
    }

    fn softmax(&self, x: &mut Array1<f32>) {
        let n = x.len();

        let x_buffer = self.create_buffer(x.as_slice().expect("x must be contiguous"));

        self.launch_softmax(&x_buffer, n);

        let result = self.read_buffer(&x_buffer, n);
        x.as_slice_mut()
            .expect("x must be contiguous")
            .copy_from_slice(&result);
    }

    fn softmax_view(&self, x: &mut ArrayViewMut1<f32>) {
        let mut owned = x.to_owned();
        self.softmax(&mut owned);
        x.assign(&owned);
    }

    fn silu(&self, x: &Array1<f32>) -> Array1<f32> {
        let n = x.len();

        let x_buffer = self.create_buffer(x.as_slice().expect("x must be contiguous"));
        let out_buffer = self.create_buffer_zeros(n);

        self.launch_silu(&x_buffer, &out_buffer, n);

        Array1::from_vec(self.read_buffer(&out_buffer, n))
    }

    fn apply_rope(&self, x: &mut Array2<f32>, cos: &Array1<f32>, sin: &Array1<f32>) {
        let (n_heads, head_dim) = x.dim();

        let x_buffer = self.create_buffer(x.as_slice().expect("x must be contiguous"));
        let cos_buffer = self.create_buffer(cos.as_slice().expect("cos must be contiguous"));
        let sin_buffer = self.create_buffer(sin.as_slice().expect("sin must be contiguous"));

        self.launch_rope(&x_buffer, &cos_buffer, &sin_buffer, n_heads, head_dim);

        let result = self.read_buffer(&x_buffer, n_heads * head_dim);
        x.as_slice_mut()
            .expect("x must be contiguous")
            .copy_from_slice(&result);
    }

    fn compute_attention_scores(
        &self,
        query: ArrayView1<f32>,
        keys: ArrayView2<f32>,
        scores: &mut ArrayViewMut1<f32>,
        scale: f32,
    ) {
        let seq_len = keys.nrows();
        let head_dim = keys.ncols();

        let query_owned = query.to_owned();
        let keys_owned = keys.to_owned();

        let query_buffer =
            self.create_buffer(query_owned.as_slice().expect("query must be contiguous"));
        let keys_buffer =
            self.create_buffer(keys_owned.as_slice().expect("keys must be contiguous"));
        let scores_buffer = self.create_buffer_zeros(seq_len);

        self.launch_attention_scores(
            &query_buffer,
            &keys_buffer,
            &scores_buffer,
            seq_len,
            head_dim,
            scale,
        );

        let result = self.read_buffer(&scores_buffer, seq_len);
        scores.assign(&Array1::from_vec(result));
    }

    fn weighted_sum_rows(
        &self,
        weights: ArrayView1<f32>,
        matrix: ArrayView2<f32>,
        out: &mut ArrayViewMut1<f32>,
    ) {
        let (n, d) = matrix.dim();

        let weights_owned = weights.to_owned();
        let matrix_owned = matrix.to_owned();

        let weights_buffer = self.create_buffer(
            weights_owned
                .as_slice()
                .expect("weights must be contiguous"),
        );
        let matrix_buffer =
            self.create_buffer(matrix_owned.as_slice().expect("matrix must be contiguous"));
        let out_buffer = self.create_buffer_zeros(d);

        self.launch_weighted_sum(&weights_buffer, &matrix_buffer, &out_buffer, n, d);

        let result = self.read_buffer(&out_buffer, d);
        out.assign(&Array1::from_vec(result));
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_webgpu_availability() {
        let available = WebGPUBackend::is_available();
        println!("WebGPU available: {}", available);
    }

    #[test]
    #[ignore]
    fn test_webgpu_backend_creation() {
        if !WebGPUBackend::is_available() {
            return;
        }

        let backend = WebGPUBackend::new().expect("Failed to create WebGPU backend");
        assert_eq!(backend.name(), "webgpu");
        println!("WebGPU device: {}", backend.device_name());
    }

    #[test]
    #[ignore]
    fn test_webgpu_matmul_vec() {
        if !WebGPUBackend::is_available() {
            return;
        }

        let backend = WebGPUBackend::new().expect("Failed to create WebGPU backend");

        let w = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0]);

        let result = backend.matmul_vec(&w, &x);

        assert_eq!(result.len(), 2);
        assert!((result[0] - 14.0).abs() < 1e-5);
        assert!((result[1] - 32.0).abs() < 1e-5);
    }

    #[test]
    #[ignore]
    fn test_webgpu_softmax() {
        if !WebGPUBackend::is_available() {
            return;
        }

        let backend = WebGPUBackend::new().expect("Failed to create WebGPU backend");

        let mut x = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        backend.softmax(&mut x);

        let sum: f32 = x.sum();
        assert!(
            (sum - 1.0).abs() < 1e-4,
            "Softmax sum should be ~1.0, got {}",
            sum
        );
    }

    #[test]
    #[ignore]
    fn test_webgpu_silu() {
        if !WebGPUBackend::is_available() {
            return;
        }

        let backend = WebGPUBackend::new().expect("Failed to create WebGPU backend");

        let x = Array1::from_vec(vec![0.0, 1.0, -1.0]);
        let result = backend.silu(&x);

        assert_eq!(result.len(), 3);
        assert!(result[0].abs() < 1e-5);
        assert!((result[1] - 0.731).abs() < 0.01);
        assert!((result[2] - (-0.269)).abs() < 0.01);
    }

    #[test]
    #[ignore]
    fn test_webgpu_rmsnorm() {
        if !WebGPUBackend::is_available() {
            return;
        }

        let backend = WebGPUBackend::new().expect("Failed to create WebGPU backend");

        let mut x = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let weight = Array1::from_vec(vec![1.0, 1.0, 1.0, 1.0]);
        let eps = 1e-5;

        backend.rmsnorm(&mut x, &weight, eps);

        let expected_rms = (30.0f32 / 4.0 + eps).sqrt();
        assert!((x[0] - 1.0 / expected_rms).abs() < 0.01);
        assert!((x[1] - 2.0 / expected_rms).abs() < 0.01);
    }
}
