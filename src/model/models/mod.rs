//! Model Architecture Implementations (Phase 8)
//!
//! This module provides concrete implementations for various LLM architectures:
//! - LLaMA (Meta)
//! - Phi (Microsoft)
//! - Gemma (Google)
//! - Qwen (Alibaba)
//! - DeepSeek (DeepSeek-AI) -- MoE architecture
//!
//! Each architecture shares common building blocks but differs in:
//! - Tensor naming conventions
//! - Activation functions
//! - Normalization approaches
//! - RoPE scaling methods
//! - Attention patterns
//! - FFN type (dense MLP vs MoE)

pub mod deepseek;
pub mod gemma;
pub mod llama;
pub mod phi;
pub mod qwen;

pub use deepseek::{DeepSeek, LazyDeepSeek};
pub use gemma::{Gemma, LazyGemma};
pub use llama::{LLaMA, LazyLLaMA};
pub use phi::{LazyPhi, Phi};
pub use qwen::{LazyQwen, Qwen};

use crate::loader::{Config, Parameters};
use crate::model::architecture::{detect_architecture_from_tensors, ModelArchitecture};
use crate::model::{InferenceState, LazyMistral, Mistral};
use anyhow::Result;

/// Dynamic model enum for runtime polymorphism
pub enum DynamicModel<'a> {
    Mistral(Mistral),
    LLaMA(LLaMA),
    Phi(Phi),
    Gemma(Gemma),
    Qwen(Qwen),
    DeepSeek(DeepSeek),
    // Lazy variants
    LazyMistral(LazyMistral<'a>),
    LazyLLaMA(LazyLLaMA<'a>),
    LazyPhi(LazyPhi<'a>),
    LazyGemma(LazyGemma<'a>),
    LazyQwen(LazyQwen<'a>),
    LazyDeepSeek(LazyDeepSeek<'a>),
}

impl DynamicModel<'_> {
    /// Load model with automatic architecture detection
    pub fn load_auto(params: Parameters) -> Result<DynamicModel<'static>> {
        let tensor_names: Vec<String> = params.tensors.keys().cloned().collect();
        let architecture = detect_architecture_from_tensors(&tensor_names);

        eprintln!("Detected architecture: {}", architecture);

        match architecture {
            ModelArchitecture::Mistral => {
                let model = Mistral::load(params)?;
                Ok(DynamicModel::Mistral(model))
            }
            ModelArchitecture::LLaMA => {
                let model = LLaMA::load(params)?;
                Ok(DynamicModel::LLaMA(model))
            }
            ModelArchitecture::Phi => {
                let model = Phi::load(params)?;
                Ok(DynamicModel::Phi(model))
            }
            ModelArchitecture::Gemma => {
                let model = Gemma::load(params)?;
                Ok(DynamicModel::Gemma(model))
            }
            ModelArchitecture::Qwen => {
                let model = Qwen::load(params)?;
                Ok(DynamicModel::Qwen(model))
            }
            ModelArchitecture::DeepSeek => {
                let model = DeepSeek::load(params)?;
                Ok(DynamicModel::DeepSeek(model))
            }
            ModelArchitecture::Unknown => {
                // Default to Mistral for unknown architectures
                eprintln!("Warning: Unknown architecture, defaulting to Mistral");
                let model = Mistral::load(params)?;
                Ok(DynamicModel::Mistral(model))
            }
        }
    }

    /// Load model with specific architecture
    pub fn load_with_arch(
        params: Parameters,
        architecture: ModelArchitecture,
    ) -> Result<DynamicModel<'static>> {
        eprintln!("Loading model as: {}", architecture);

        match architecture {
            ModelArchitecture::Mistral => {
                let model = Mistral::load(params)?;
                Ok(DynamicModel::Mistral(model))
            }
            ModelArchitecture::LLaMA => {
                let model = LLaMA::load(params)?;
                Ok(DynamicModel::LLaMA(model))
            }
            ModelArchitecture::Phi => {
                let model = Phi::load(params)?;
                Ok(DynamicModel::Phi(model))
            }
            ModelArchitecture::Gemma => {
                let model = Gemma::load(params)?;
                Ok(DynamicModel::Gemma(model))
            }
            ModelArchitecture::Qwen => {
                let model = Qwen::load(params)?;
                Ok(DynamicModel::Qwen(model))
            }
            ModelArchitecture::DeepSeek => {
                let model = DeepSeek::load(params)?;
                Ok(DynamicModel::DeepSeek(model))
            }
            ModelArchitecture::Unknown => {
                anyhow::bail!("Cannot load model with unknown architecture")
            }
        }
    }

    /// Get the architecture type
    pub fn architecture(&self) -> ModelArchitecture {
        match self {
            DynamicModel::Mistral(_) | DynamicModel::LazyMistral(_) => ModelArchitecture::Mistral,
            DynamicModel::LLaMA(_) | DynamicModel::LazyLLaMA(_) => ModelArchitecture::LLaMA,
            DynamicModel::Phi(_) | DynamicModel::LazyPhi(_) => ModelArchitecture::Phi,
            DynamicModel::Gemma(_) | DynamicModel::LazyGemma(_) => ModelArchitecture::Gemma,
            DynamicModel::Qwen(_) | DynamicModel::LazyQwen(_) => ModelArchitecture::Qwen,
            DynamicModel::DeepSeek(_) | DynamicModel::LazyDeepSeek(_) => {
                ModelArchitecture::DeepSeek
            }
        }
    }

    /// Get the model config
    pub fn config(&self) -> &Config {
        match self {
            DynamicModel::Mistral(m) => &m.config,
            DynamicModel::LLaMA(m) => &m.config,
            DynamicModel::Phi(m) => &m.config,
            DynamicModel::Gemma(m) => &m.config,
            DynamicModel::Qwen(m) => &m.config,
            DynamicModel::DeepSeek(m) => &m.config,
            DynamicModel::LazyMistral(m) => &m.config,
            DynamicModel::LazyLLaMA(m) => &m.config,
            DynamicModel::LazyPhi(m) => &m.config,
            DynamicModel::LazyGemma(m) => &m.config,
            DynamicModel::LazyQwen(m) => &m.config,
            DynamicModel::LazyDeepSeek(m) => &m.config,
        }
    }

    /// Run forward pass
    pub fn forward(&self, state: &mut InferenceState, token: u32, debug: bool) {
        match self {
            DynamicModel::Mistral(m) => m.forward(state, token, debug),
            DynamicModel::LLaMA(m) => m.forward(state, token, debug),
            DynamicModel::Phi(m) => m.forward(state, token, debug),
            DynamicModel::Gemma(m) => m.forward(state, token, debug),
            DynamicModel::Qwen(m) => m.forward(state, token, debug),
            DynamicModel::DeepSeek(m) => m.forward(state, token, debug),
            DynamicModel::LazyMistral(m) => m.forward(state, token, debug),
            DynamicModel::LazyLLaMA(m) => m.forward(state, token, debug),
            DynamicModel::LazyPhi(m) => m.forward(state, token, debug),
            DynamicModel::LazyGemma(m) => m.forward(state, token, debug),
            DynamicModel::LazyQwen(m) => m.forward(state, token, debug),
            DynamicModel::LazyDeepSeek(m) => m.forward(state, token, debug),
        }
    }

    /// Run optimized forward pass
    pub fn fast_forward(&self, state: &mut InferenceState, token: u32, debug: bool) {
        match self {
            DynamicModel::Mistral(m) => m.fast_forward(state, token, debug),
            DynamicModel::LLaMA(m) => m.fast_forward(state, token, debug),
            DynamicModel::Phi(m) => m.fast_forward(state, token, debug),
            DynamicModel::Gemma(m) => m.fast_forward(state, token, debug),
            DynamicModel::Qwen(m) => m.fast_forward(state, token, debug),
            DynamicModel::DeepSeek(m) => m.fast_forward(state, token, debug),
            DynamicModel::LazyMistral(m) => m.fast_forward(state, token, debug),
            DynamicModel::LazyLLaMA(m) => m.fast_forward(state, token, debug),
            DynamicModel::LazyPhi(m) => m.fast_forward(state, token, debug),
            DynamicModel::LazyGemma(m) => m.fast_forward(state, token, debug),
            DynamicModel::LazyQwen(m) => m.fast_forward(state, token, debug),
            DynamicModel::LazyDeepSeek(m) => m.fast_forward(state, token, debug),
        }
    }

    /// Encode text to tokens
    pub fn encode(&self, text: &str) -> Vec<u32> {
        match self {
            DynamicModel::Mistral(m) => m.tokenizer.encode(text),
            DynamicModel::LLaMA(m) => m.tokenizer.encode(text),
            DynamicModel::Phi(m) => m.tokenizer.encode(text),
            DynamicModel::Gemma(m) => m.tokenizer.encode(text),
            DynamicModel::Qwen(m) => m.tokenizer.encode(text),
            DynamicModel::DeepSeek(m) => m.tokenizer.encode(text),
            DynamicModel::LazyMistral(m) => m.tokenizer.encode(text),
            DynamicModel::LazyLLaMA(m) => m.tokenizer.encode(text),
            DynamicModel::LazyPhi(m) => m.tokenizer.encode(text),
            DynamicModel::LazyGemma(m) => m.tokenizer.encode(text),
            DynamicModel::LazyQwen(m) => m.tokenizer.encode(text),
            DynamicModel::LazyDeepSeek(m) => m.tokenizer.encode(text),
        }
    }

    /// Decode tokens to text
    pub fn decode(&self, tokens: &[u32]) -> String {
        match self {
            DynamicModel::Mistral(m) => m.tokenizer.decode(tokens),
            DynamicModel::LLaMA(m) => m.tokenizer.decode(tokens),
            DynamicModel::Phi(m) => m.tokenizer.decode(tokens),
            DynamicModel::Gemma(m) => m.tokenizer.decode(tokens),
            DynamicModel::Qwen(m) => m.tokenizer.decode(tokens),
            DynamicModel::DeepSeek(m) => m.tokenizer.decode(tokens),
            DynamicModel::LazyMistral(m) => m.tokenizer.decode(tokens),
            DynamicModel::LazyLLaMA(m) => m.tokenizer.decode(tokens),
            DynamicModel::LazyPhi(m) => m.tokenizer.decode(tokens),
            DynamicModel::LazyGemma(m) => m.tokenizer.decode(tokens),
            DynamicModel::LazyQwen(m) => m.tokenizer.decode(tokens),
            DynamicModel::LazyDeepSeek(m) => m.tokenizer.decode(tokens),
        }
    }

    /// Get logits
    pub fn get_logits<'b>(&self, state: &'b InferenceState) -> &'b [f32] {
        state.logits.as_slice().unwrap()
    }
}

/// Model loader that auto-detects architecture
pub struct ModelLoader;

impl ModelLoader {
    /// Load model with automatic architecture detection
    pub fn load_auto(path: &str) -> Result<DynamicModel<'static>> {
        let params = Parameters::load(path)?;
        DynamicModel::load_auto(params)
    }

    /// Load model as specific architecture
    pub fn load_as(path: &str, arch: ModelArchitecture) -> Result<DynamicModel<'static>> {
        let params = Parameters::load(path)?;
        DynamicModel::load_with_arch(params, arch)
    }
}
