//! Speculative Decoding Implementation
//!
//! Speculative decoding accelerates text generation by using a smaller, faster
//! "draft" model to propose multiple candidate tokens, then verifying them in
//! parallel with the main model.
//!
//! # Algorithm
//! 1. Draft model generates K candidate tokens autoregressively
//! 2. Main model processes all K+1 positions in a single forward pass
//! 3. Tokens are accepted until the first rejection (verification fails)
//! 4. On rejection, the main model's distribution at that position is used
//!
//! # Performance
//! - Speed improvement: 2-3x for well-matched draft/main model pairs
//! - Quality: Mathematically equivalent to main model alone
//!
//! # References
//! - Fast Inference from Transformers via Speculative Decoding (Leviathan et al., 2023)
//! - Accelerating Large Language Model Decoding with Speculative Sampling (Chen et al., 2023)

use crate::model::InferenceState;
use rand::Rng;
use std::collections::VecDeque;

/// Configuration for speculative decoding.
#[derive(Debug, Clone)]
pub struct SpeculativeConfig {
    /// Number of tokens to speculate (propose) per iteration
    pub speculation_length: usize,
    /// Temperature for sampling (0 = greedy)
    pub temperature: f32,
    /// Minimum probability ratio for acceptance (rejection sampling threshold)
    pub min_p_ratio: f32,
    /// Whether to use adaptive speculation length
    pub adaptive: bool,
    /// Minimum speculation length when adaptive
    pub min_speculation: usize,
    /// Maximum speculation length when adaptive
    pub max_speculation: usize,
}

impl Default for SpeculativeConfig {
    fn default() -> Self {
        Self {
            speculation_length: 5,
            temperature: 0.7,
            min_p_ratio: 0.0,
            adaptive: true,
            min_speculation: 2,
            max_speculation: 8,
        }
    }
}

impl SpeculativeConfig {
    /// Create a config optimized for quality (lower speculation, more conservative)
    pub fn quality() -> Self {
        Self {
            speculation_length: 3,
            temperature: 0.0, // Greedy
            min_p_ratio: 0.0,
            adaptive: false,
            min_speculation: 3,
            max_speculation: 3,
        }
    }

    /// Create a config optimized for speed (higher speculation)
    pub fn speed() -> Self {
        Self {
            speculation_length: 8,
            temperature: 0.7,
            min_p_ratio: 0.0,
            adaptive: true,
            min_speculation: 4,
            max_speculation: 12,
        }
    }
}

/// Statistics from speculative decoding for monitoring performance.
#[derive(Debug, Clone, Default)]
pub struct SpeculativeStats {
    /// Total tokens generated
    pub tokens_generated: usize,
    /// Number of speculative iterations
    pub iterations: usize,
    /// Number of draft tokens accepted
    pub tokens_accepted: usize,
    /// Number of draft tokens rejected
    pub tokens_rejected: usize,
    /// Current adaptive speculation length
    pub current_speculation_length: usize,
}

impl SpeculativeStats {
    /// Get the acceptance rate
    pub fn acceptance_rate(&self) -> f32 {
        let total = self.tokens_accepted + self.tokens_rejected;
        if total == 0 {
            1.0
        } else {
            self.tokens_accepted as f32 / total as f32
        }
    }

    /// Get the average tokens per iteration (speedup indicator)
    pub fn tokens_per_iteration(&self) -> f32 {
        if self.iterations == 0 {
            0.0
        } else {
            self.tokens_generated as f32 / self.iterations as f32
        }
    }
}

/// Trait for models that can be used as draft or main models in speculative decoding.
pub trait SpeculativeModel {
    /// Run forward pass for a single token and return next token probabilities.
    fn forward(&self, state: &mut InferenceState, token: u32);

    /// Get the vocabulary size.
    fn vocab_size(&self) -> usize;

    /// Get the current logits after a forward pass.
    fn get_logits<'a>(&self, state: &'a InferenceState) -> &'a [f32];

    /// Clone the KV cache state (for speculative rollback).
    fn save_cache_state(&self, state: &InferenceState) -> CacheState;

    /// Restore the KV cache state.
    fn restore_cache_state(&self, state: &mut InferenceState, cache: &CacheState);
}

/// Saved KV cache state for rollback during speculation.
#[derive(Clone)]
pub struct CacheState {
    /// Position in the sequence
    pub pos: usize,
    // Note: For full rollback support, we would need to clone k_cache and v_cache.
    // For now, we use a simpler approach where we track position and re-run if needed.
}

/// Speculative decoder that uses a draft model to accelerate generation.
///
/// # Type Parameters
/// * `D` - Draft model type (smaller, faster model)
/// * `M` - Main model type (larger, more accurate model)
pub struct SpeculativeDecoder<'a> {
    /// Main model for verification
    main_forward: Box<dyn Fn(&mut InferenceState, u32) + 'a>,
    /// Draft model for speculation
    draft_forward: Box<dyn Fn(&mut InferenceState, u32) + 'a>,
    /// Configuration
    config: SpeculativeConfig,
    /// Statistics
    stats: SpeculativeStats,
}

impl<'a> SpeculativeDecoder<'a> {
    /// Create a new speculative decoder.
    ///
    /// # Arguments
    /// * `main_forward` - Function to run main model forward pass
    /// * `draft_forward` - Function to run draft model forward pass
    /// * `config` - Speculative decoding configuration
    pub fn new<MF, DF>(main_forward: MF, draft_forward: DF, config: SpeculativeConfig) -> Self
    where
        MF: Fn(&mut InferenceState, u32) + 'a,
        DF: Fn(&mut InferenceState, u32) + 'a,
    {
        Self {
            main_forward: Box::new(main_forward),
            draft_forward: Box::new(draft_forward),
            config,
            stats: SpeculativeStats::default(),
        }
    }

    /// Get current statistics.
    pub fn stats(&self) -> &SpeculativeStats {
        &self.stats
    }

    /// Reset statistics.
    pub fn reset_stats(&mut self) {
        self.stats = SpeculativeStats::default();
    }

    /// Generate one iteration of speculative decoding.
    ///
    /// Returns a vector of accepted tokens for this iteration.
    ///
    /// # Arguments
    /// * `main_state` - State for the main model
    /// * `draft_state` - State for the draft model
    /// * `context_token` - The last token to continue from
    pub fn generate_step(
        &mut self,
        main_state: &mut InferenceState,
        draft_state: &mut InferenceState,
        context_token: u32,
    ) -> Vec<u32> {
        let mut rng = rand::thread_rng();
        let speculation_length = if self.config.adaptive {
            self.stats
                .current_speculation_length
                .max(self.config.min_speculation)
        } else {
            self.config.speculation_length
        };

        // Phase 1: Draft model generates K candidate tokens
        let mut draft_tokens = Vec::with_capacity(speculation_length);
        let mut draft_probs = Vec::with_capacity(speculation_length);

        let mut current_token = context_token;
        for _ in 0..speculation_length {
            (self.draft_forward)(draft_state, current_token);

            // Get probabilities from draft model
            let probs = softmax_with_temperature(
                &draft_state.logits.as_slice().unwrap(),
                self.config.temperature,
            );

            // Sample next token
            let next_token = sample_from_probs(&probs, &mut rng);
            let prob = probs[next_token as usize];

            draft_tokens.push(next_token);
            draft_probs.push((next_token, prob));
            current_token = next_token;
            draft_state.pos += 1;
        }

        // Phase 2: Main model verifies all tokens
        // Run main model on context_token and all draft tokens
        let mut accepted_tokens = Vec::new();
        let mut main_token = context_token;

        for i in 0..draft_tokens.len() {
            (self.main_forward)(main_state, main_token);

            // Get main model probabilities
            let main_probs = softmax_with_temperature(
                &main_state.logits.as_slice().unwrap(),
                self.config.temperature,
            );

            // Get the draft token and its probability under both models
            let draft_token = draft_tokens[i];
            let (_, draft_prob) = draft_probs[i];
            let main_prob = main_probs[draft_token as usize];

            // Rejection sampling: accept with probability min(1, main_prob / draft_prob)
            let accept_prob = if draft_prob > 0.0 {
                (main_prob / draft_prob).min(1.0)
            } else {
                0.0
            };

            let random_val: f32 = rng.gen();

            if random_val < accept_prob {
                // Accept the draft token
                accepted_tokens.push(draft_token);
                main_state.pos += 1;
                main_token = draft_token;
                self.stats.tokens_accepted += 1;
            } else {
                // Reject: sample from adjusted distribution
                let adjusted_probs = adjust_distribution(
                    &main_probs,
                    &softmax_with_temperature(
                        &draft_state.logits.as_slice().unwrap(),
                        self.config.temperature,
                    ),
                );
                let corrected_token = sample_from_probs(&adjusted_probs, &mut rng);
                accepted_tokens.push(corrected_token);
                main_state.pos += 1;
                self.stats.tokens_rejected += 1;

                // Don't continue verifying after rejection
                break;
            }
        }

        // If all draft tokens were accepted, sample one more from main model
        if accepted_tokens.len() == draft_tokens.len() {
            (self.main_forward)(
                main_state,
                *accepted_tokens.last().unwrap_or(&context_token),
            );
            let main_probs = softmax_with_temperature(
                &main_state.logits.as_slice().unwrap(),
                self.config.temperature,
            );
            let bonus_token = sample_from_probs(&main_probs, &mut rng);
            accepted_tokens.push(bonus_token);
            main_state.pos += 1;
        }

        // Update statistics
        self.stats.tokens_generated += accepted_tokens.len();
        self.stats.iterations += 1;

        // Adaptive speculation length adjustment
        if self.config.adaptive {
            let acceptance_rate = self.stats.acceptance_rate();
            if acceptance_rate > 0.9
                && self.stats.current_speculation_length < self.config.max_speculation
            {
                self.stats.current_speculation_length += 1;
            } else if acceptance_rate < 0.5
                && self.stats.current_speculation_length > self.config.min_speculation
            {
                self.stats.current_speculation_length -= 1;
            }
        }

        // Sync draft state position with main state
        draft_state.pos = main_state.pos;

        accepted_tokens
    }
}

/// Simplified speculative generation for a single model (self-speculative).
///
/// This uses the same model at different temperatures or with early-exit
/// to generate draft tokens. Less effective than true two-model speculative
/// decoding, but requires only one model.
pub struct SelfSpeculativeDecoder<'a> {
    /// Model forward function
    forward: Box<dyn Fn(&mut InferenceState, u32) + 'a>,
    /// Configuration
    config: SpeculativeConfig,
    /// Draft temperature (higher = faster but less accurate)
    draft_temperature: f32,
    /// Main temperature
    main_temperature: f32,
    /// Statistics
    stats: SpeculativeStats,
}

impl<'a> SelfSpeculativeDecoder<'a> {
    /// Create a self-speculative decoder.
    pub fn new<F>(
        forward: F,
        config: SpeculativeConfig,
        draft_temperature: f32,
        main_temperature: f32,
    ) -> Self
    where
        F: Fn(&mut InferenceState, u32) + 'a,
    {
        Self {
            forward: Box::new(forward),
            config,
            draft_temperature,
            main_temperature,
            stats: SpeculativeStats::default(),
        }
    }

    /// Generate tokens using self-speculative decoding.
    pub fn generate_step(&mut self, state: &mut InferenceState, context_token: u32) -> Vec<u32> {
        let mut rng = rand::thread_rng();
        let speculation_length = self.config.speculation_length;

        // Save initial position
        let start_pos = state.pos;

        // Phase 1: Generate draft tokens with higher temperature (more random)
        let mut draft_tokens = Vec::with_capacity(speculation_length);
        let mut draft_probs = Vec::with_capacity(speculation_length);

        let mut current_token = context_token;
        for _ in 0..speculation_length {
            (self.forward)(state, current_token);

            let probs =
                softmax_with_temperature(&state.logits.as_slice().unwrap(), self.draft_temperature);
            let next_token = sample_from_probs(&probs, &mut rng);
            let prob = probs[next_token as usize];

            draft_tokens.push(next_token);
            draft_probs.push(prob);
            current_token = next_token;
            state.pos += 1;
        }

        // Phase 2: Verify with main temperature
        // Reset to start position
        state.pos = start_pos;

        let mut accepted_tokens = Vec::new();
        current_token = context_token;

        for i in 0..draft_tokens.len() {
            (self.forward)(state, current_token);

            let main_probs =
                softmax_with_temperature(&state.logits.as_slice().unwrap(), self.main_temperature);
            let draft_token = draft_tokens[i];
            let draft_prob = draft_probs[i];
            let main_prob = main_probs[draft_token as usize];

            // Acceptance probability
            let accept_prob = if draft_prob > 0.0 {
                (main_prob / draft_prob).min(1.0)
            } else {
                0.0
            };

            if rng.gen::<f32>() < accept_prob {
                accepted_tokens.push(draft_token);
                state.pos += 1;
                current_token = draft_token;
                self.stats.tokens_accepted += 1;
            } else {
                // Sample from main distribution
                let corrected_token = sample_from_probs(&main_probs, &mut rng);
                accepted_tokens.push(corrected_token);
                state.pos += 1;
                self.stats.tokens_rejected += 1;
                break;
            }
        }

        // Bonus token if all accepted
        if accepted_tokens.len() == draft_tokens.len() {
            (self.forward)(state, *accepted_tokens.last().unwrap_or(&context_token));
            let main_probs =
                softmax_with_temperature(&state.logits.as_slice().unwrap(), self.main_temperature);
            let bonus_token = sample_from_probs(&main_probs, &mut rng);
            accepted_tokens.push(bonus_token);
            state.pos += 1;
        }

        self.stats.tokens_generated += accepted_tokens.len();
        self.stats.iterations += 1;

        accepted_tokens
    }

    /// Get statistics.
    pub fn stats(&self) -> &SpeculativeStats {
        &self.stats
    }
}

/// Lookahead decoding: predict multiple future tokens in parallel.
///
/// This is a simpler form of speculation that predicts the most likely
/// next K tokens and verifies them.
pub struct LookaheadDecoder {
    /// Number of tokens to look ahead
    pub lookahead: usize,
    /// Temperature for sampling
    pub temperature: f32,
}

impl LookaheadDecoder {
    pub fn new(lookahead: usize, temperature: f32) -> Self {
        Self {
            lookahead,
            temperature,
        }
    }

    /// Generate greedy lookahead tokens (returns top-k most likely sequences).
    pub fn generate_candidates(&self, logits: &[f32], top_k: usize) -> Vec<(u32, f32)> {
        let probs = softmax_with_temperature(logits, self.temperature);

        // Get top-k tokens and their probabilities
        let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        indexed
            .into_iter()
            .take(top_k)
            .map(|(idx, prob)| (idx as u32, prob))
            .collect()
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Apply softmax with temperature scaling.
fn softmax_with_temperature(logits: &[f32], temperature: f32) -> Vec<f32> {
    let temp = if temperature <= 0.0 {
        1e-8
    } else {
        temperature
    };

    // Scale by temperature
    let scaled: Vec<f32> = logits.iter().map(|&x| x / temp).collect();

    // Find max for numerical stability
    let max_val = scaled.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    // Compute exp and sum
    let mut probs: Vec<f32> = scaled.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f32 = probs.iter().sum();

    // Normalize
    let inv_sum = 1.0 / sum;
    for p in probs.iter_mut() {
        *p *= inv_sum;
    }

    probs
}

/// Sample a token from a probability distribution.
fn sample_from_probs<R: Rng>(probs: &[f32], rng: &mut R) -> u32 {
    let r: f32 = rng.gen();
    let mut cumsum = 0.0;

    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if cumsum >= r {
            return i as u32;
        }
    }

    (probs.len() - 1) as u32
}

/// Adjust distribution for rejection sampling correction.
/// Returns max(0, main_probs - draft_probs) normalized.
fn adjust_distribution(main_probs: &[f32], draft_probs: &[f32]) -> Vec<f32> {
    let mut adjusted: Vec<f32> = main_probs
        .iter()
        .zip(draft_probs.iter())
        .map(|(&m, &d)| (m - d).max(0.0))
        .collect();

    let sum: f32 = adjusted.iter().sum();
    if sum > 0.0 {
        let inv_sum = 1.0 / sum;
        for p in adjusted.iter_mut() {
            *p *= inv_sum;
        }
    } else {
        // Fallback to main distribution if adjustment is all zeros
        adjusted.copy_from_slice(main_probs);
    }

    adjusted
}

/// Token buffer for managing generated tokens with rollback support.
#[derive(Debug, Clone)]
pub struct TokenBuffer {
    /// Committed tokens (verified and accepted)
    committed: Vec<u32>,
    /// Pending tokens (speculated but not yet verified)
    pending: VecDeque<u32>,
}

impl TokenBuffer {
    pub fn new() -> Self {
        Self {
            committed: Vec::new(),
            pending: VecDeque::new(),
        }
    }

    /// Add a committed token.
    pub fn commit(&mut self, token: u32) {
        self.committed.push(token);
    }

    /// Add pending (speculated) tokens.
    pub fn add_pending(&mut self, tokens: &[u32]) {
        for &t in tokens {
            self.pending.push_back(t);
        }
    }

    /// Accept N pending tokens.
    pub fn accept(&mut self, n: usize) {
        for _ in 0..n.min(self.pending.len()) {
            if let Some(t) = self.pending.pop_front() {
                self.committed.push(t);
            }
        }
    }

    /// Reject all pending tokens.
    pub fn reject_all(&mut self) {
        self.pending.clear();
    }

    /// Get all committed tokens.
    pub fn tokens(&self) -> &[u32] {
        &self.committed
    }

    /// Get the last committed token.
    pub fn last(&self) -> Option<u32> {
        self.committed.last().copied()
    }

    /// Total committed length.
    pub fn len(&self) -> usize {
        self.committed.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.committed.is_empty()
    }
}

impl Default for TokenBuffer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax_with_temperature() {
        let logits = vec![1.0, 2.0, 3.0, 4.0];

        // Temperature 1.0 should give standard softmax
        let probs = softmax_with_temperature(&logits, 1.0);
        assert!((probs.iter().sum::<f32>() - 1.0).abs() < 1e-6);

        // Higher temperature should make distribution more uniform
        let probs_high = softmax_with_temperature(&logits, 2.0);
        let entropy_high: f32 = probs_high
            .iter()
            .map(|&p| if p > 0.0 { -p * p.ln() } else { 0.0 })
            .sum();
        let entropy_low: f32 = probs
            .iter()
            .map(|&p| if p > 0.0 { -p * p.ln() } else { 0.0 })
            .sum();
        assert!(entropy_high > entropy_low);

        // Very low temperature should approach one-hot
        let probs_cold = softmax_with_temperature(&logits, 0.01);
        assert!(probs_cold[3] > 0.99); // Highest logit should dominate
    }

    #[test]
    fn test_sample_from_probs() {
        let probs = vec![0.0, 0.0, 1.0, 0.0]; // Deterministic
        let mut rng = rand::thread_rng();

        for _ in 0..10 {
            let token = sample_from_probs(&probs, &mut rng);
            assert_eq!(token, 2);
        }
    }

    #[test]
    fn test_adjust_distribution() {
        let main = vec![0.3, 0.3, 0.2, 0.2];
        let draft = vec![0.1, 0.4, 0.3, 0.2];

        let adjusted = adjust_distribution(&main, &draft);

        // Should sum to 1
        assert!((adjusted.iter().sum::<f32>() - 1.0).abs() < 1e-6);

        // main[0] > draft[0], so adjusted[0] > 0
        assert!(adjusted[0] > 0.0);

        // main[1] < draft[1], so adjusted[1] = 0
        assert!(adjusted[1].abs() < 1e-6);
    }

    #[test]
    fn test_token_buffer() {
        let mut buffer = TokenBuffer::new();

        buffer.commit(1);
        buffer.commit(2);
        assert_eq!(buffer.len(), 2);
        assert_eq!(buffer.last(), Some(2));

        buffer.add_pending(&[3, 4, 5]);
        assert_eq!(buffer.len(), 2); // Only committed counts

        buffer.accept(2);
        assert_eq!(buffer.len(), 4);
        assert_eq!(buffer.tokens(), &[1, 2, 3, 4]);

        buffer.reject_all();
        assert_eq!(buffer.len(), 4); // Committed unchanged
    }

    #[test]
    fn test_speculative_stats() {
        let mut stats = SpeculativeStats::default();

        stats.tokens_accepted = 8;
        stats.tokens_rejected = 2;
        stats.tokens_generated = 12;
        stats.iterations = 3;

        assert!((stats.acceptance_rate() - 0.8).abs() < 1e-6);
        assert!((stats.tokens_per_iteration() - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_lookahead_decoder() {
        let decoder = LookaheadDecoder::new(4, 1.0);
        let logits = vec![1.0, 5.0, 2.0, 8.0, 3.0];

        let candidates = decoder.generate_candidates(&logits, 3);

        assert_eq!(candidates.len(), 3);
        assert_eq!(candidates[0].0, 3); // Highest logit
        assert!(candidates[0].1 > candidates[1].1); // Sorted by probability
    }
}
