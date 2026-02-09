mod attention;
mod embedding;
pub mod flash_attention;
mod layer;
mod lazy_attention;
mod lazy_embedding;
mod lazy_layer;
mod lazy_mlp;
mod mlp;
mod rmsnorm;
#[cfg(test)]
mod tests;

pub use attention::Attention;
pub use embedding::Embedding;
pub use layer::Layer;
pub use mlp::MLP;
pub use rmsnorm::RMSNorm;

// Lazy loading variants
pub use lazy_attention::LazyAttention;
pub use lazy_embedding::LazyEmbedding;
pub use lazy_layer::LazyLayer;
pub use lazy_mlp::LazyMLP;

// Flash Attention (Phase 4 Algorithmic Optimization)
pub use flash_attention::{
    estimate_memory as flash_attention_estimate_memory, flash_attention_into,
    flash_attention_multi_head, flash_attention_single_head, FlashAttentionConfig,
    DEFAULT_TILE_SIZE as FLASH_TILE_SIZE, FLASH_ATTENTION_THRESHOLD,
};

#[cfg(feature = "parallel")]
pub use flash_attention::flash_attention_parallel;
