use crate::model::InferenceState;
use rand::Rng;

#[cfg(test)]
mod tests;

// =============================================================================
// Sampling Configuration
// =============================================================================

/// Configuration for token sampling strategies.
///
/// Controls temperature scaling, top-k filtering, and nucleus (top-p) sampling.
/// When both `top_k` and `top_p` are set, they are applied in order:
/// temperature → top-k → softmax → top-p → sample.
#[derive(Clone, Debug)]
pub struct SamplingConfig {
    /// Temperature for logit scaling. 0.0 = greedy, 1.0 = unmodified, >1.0 = more random.
    pub temperature: f32,
    /// If set, only the top K tokens by logit value are considered. The rest are masked out.
    pub top_k: Option<usize>,
    /// If set, only the smallest set of tokens whose cumulative probability >= p are kept.
    pub top_p: Option<f32>,
}

impl SamplingConfig {
    /// Greedy sampling (always pick the highest-probability token).
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_k: None,
            top_p: None,
        }
    }

    /// Sample with the given temperature, no top-k or top-p filtering.
    pub fn with_temperature(temperature: f32) -> Self {
        Self {
            temperature,
            top_k: None,
            top_p: None,
        }
    }
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: None,
            top_p: None,
        }
    }
}

// =============================================================================
// Core Sampling Functions
// =============================================================================

pub fn sample_greedy(state: &InferenceState) -> u32 {
    state
        .logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(idx, _)| idx as u32)
        .unwrap_or(0)
}

pub fn sample_multinomial(state: &mut InferenceState, temperature: f32) -> u32 {
    if temperature <= 0.0 {
        return sample_greedy(state);
    }

    // Copy logits to probs and apply temperature
    state.probs.assign(&state.logits);
    state.probs.mapv_inplace(|v| v / temperature);

    // Softmax
    crate::kernels::softmax(&mut state.probs);

    // Sample from distribution
    sample_from_probs(state)
}

// =============================================================================
// Top-K Sampling
// =============================================================================

/// Apply top-k filtering to logits (before softmax).
///
/// Keeps only the `k` largest logit values and sets the rest to negative infinity.
fn apply_top_k(probs: &mut ndarray::Array1<f32>, k: usize) {
    // Find the k-th largest value as the threshold
    let mut sorted: Vec<f32> = probs.iter().copied().collect();
    sorted.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap());
    let threshold = sorted[k.min(sorted.len()) - 1];

    // Count how many values are >= threshold. If more than k, we need to be
    // more selective (handle ties at the boundary).
    let count_above = probs.iter().filter(|&&v| v >= threshold).count();

    if count_above <= k {
        // Simple case: mask out everything below threshold
        probs.mapv_inplace(|v| if v < threshold { f32::NEG_INFINITY } else { v });
    } else {
        // Tie-breaking: there are more values equal to the threshold than we want.
        // Keep all values strictly above threshold, then keep enough at-threshold
        // values to reach exactly k.
        let mut kept = 0;
        for v in probs.iter_mut() {
            if *v >= threshold && kept < k {
                kept += 1;
            } else {
                *v = f32::NEG_INFINITY;
            }
        }
    }
}

/// Sample a token using top-k filtering with temperature.
///
/// Only the top `k` tokens by logit value are considered. The rest are masked
/// out before softmax and sampling.
pub fn sample_top_k(state: &mut InferenceState, temperature: f32, k: usize) -> u32 {
    sample_with_config(
        state,
        &SamplingConfig {
            temperature,
            top_k: Some(k),
            top_p: None,
        },
    )
}

// =============================================================================
// Top-P (Nucleus) Sampling
// =============================================================================

/// Apply top-p (nucleus) filtering to a probability distribution (after softmax).
///
/// Sorts tokens by probability descending, keeps the smallest set whose
/// cumulative probability is >= `p`, zeros out the rest, and re-normalizes.
fn apply_top_p(probs: &mut ndarray::Array1<f32>, p: f32) {
    // Create index-probability pairs and sort descending by probability
    let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
    indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Find how many tokens are needed to reach cumulative probability >= p
    let mut cumsum = 0.0;
    let mut cutoff = indexed.len();
    for (i, &(_, prob)) in indexed.iter().enumerate() {
        cumsum += prob;
        if cumsum >= p {
            cutoff = i + 1;
            break;
        }
    }

    // Zero out everything beyond the cutoff
    for &(idx, _) in &indexed[cutoff..] {
        probs[idx] = 0.0;
    }

    // Re-normalize so probabilities sum to 1
    let sum: f32 = probs.iter().sum();
    if sum > 0.0 && (sum - 1.0).abs() > 1e-7 {
        probs.mapv_inplace(|v| v / sum);
    }
}

/// Sample a token using nucleus (top-p) sampling with temperature.
///
/// After applying temperature and softmax, only the smallest set of tokens
/// whose cumulative probability >= `p` are kept.
pub fn sample_top_p(state: &mut InferenceState, temperature: f32, p: f32) -> u32 {
    sample_with_config(
        state,
        &SamplingConfig {
            temperature,
            top_k: None,
            top_p: Some(p),
        },
    )
}

// =============================================================================
// Unified Config-Based Sampling
// =============================================================================

/// Sample a token using the full sampling pipeline configured by `SamplingConfig`.
///
/// Applies strategies in order: temperature → top-k → softmax → top-p → sample.
pub fn sample_with_config(state: &mut InferenceState, config: &SamplingConfig) -> u32 {
    if config.temperature <= 0.0 {
        return sample_greedy(state);
    }

    // Copy logits to probs and apply temperature
    state.probs.assign(&state.logits);
    state.probs.mapv_inplace(|v| v / config.temperature);

    // Apply top-k filtering on logits (before softmax)
    if let Some(k) = config.top_k {
        if k > 0 && k < state.probs.len() {
            apply_top_k(&mut state.probs, k);
        }
    }

    // Softmax
    crate::kernels::softmax(&mut state.probs);

    // Apply top-p filtering on probabilities (after softmax)
    if let Some(p) = config.top_p {
        if p > 0.0 && p < 1.0 {
            apply_top_p(&mut state.probs, p);
        }
    }

    // Sample from distribution
    sample_from_probs(state)
}

/// Internal helper: sample a token from the probability distribution in `state.probs`.
fn sample_from_probs(state: &InferenceState) -> u32 {
    let mut rng = rand::thread_rng();
    let r: f32 = rng.gen();
    let mut cumsum = 0.0;

    for (i, &prob) in state.probs.iter().enumerate() {
        cumsum += prob;
        if cumsum >= r {
            return i as u32;
        }
    }

    (state.probs.len() - 1) as u32
}

// =============================================================================
// Generation Functions
// =============================================================================

/// Generate one token using the model.
pub fn generate(
    model: &crate::Mistral,
    state: &mut InferenceState,
    token: u32,
    config: &SamplingConfig,
    debug: bool,
) -> u32 {
    model.fast_forward(state, token, debug);
    sample_with_config(state, config)
}

/// Result of EOS-aware generation.
pub struct GenerationResult {
    /// The generated tokens (excluding the EOS token, if one was produced).
    pub tokens: Vec<u32>,
    /// Whether generation stopped due to an EOS token (vs. hitting max_tokens).
    pub stopped_at_eos: bool,
}

/// Generate tokens until an EOS token is produced or `max_tokens` is reached.
///
/// Calls `on_token` for each generated token (for streaming output).
/// The `eos_token_ids` slice lists all token IDs that should stop generation.
#[allow(clippy::too_many_arguments)]
pub fn generate_until_eos<F>(
    model: &crate::Mistral,
    state: &mut InferenceState,
    first_token: u32,
    config: &SamplingConfig,
    max_tokens: usize,
    eos_token_ids: &[u32],
    debug: bool,
    mut on_token: F,
) -> GenerationResult
where
    F: FnMut(u32),
{
    let mut tokens = Vec::new();
    let mut token = first_token;

    for _ in 0..max_tokens {
        token = generate(model, state, token, config, debug);

        // Check for EOS
        if eos_token_ids.contains(&token) {
            return GenerationResult {
                tokens,
                stopped_at_eos: true,
            };
        }

        on_token(token);
        tokens.push(token);
        state.pos += 1;
    }

    GenerationResult {
        tokens,
        stopped_at_eos: false,
    }
}

/// Generate one token using the lazy model (memory-efficient).
pub fn generate_lazy(
    model: &crate::LazyMistral,
    state: &mut InferenceState,
    token: u32,
    config: &SamplingConfig,
    debug: bool,
) -> u32 {
    model.fast_forward(state, token, debug);
    sample_with_config(state, config)
}

/// Generate tokens using the lazy model until EOS or `max_tokens` is reached.
///
/// Same as `generate_until_eos` but uses `LazyMistral` for memory-efficient inference.
#[allow(clippy::too_many_arguments)]
pub fn generate_lazy_until_eos<F>(
    model: &crate::LazyMistral,
    state: &mut InferenceState,
    first_token: u32,
    config: &SamplingConfig,
    max_tokens: usize,
    eos_token_ids: &[u32],
    debug: bool,
    mut on_token: F,
) -> GenerationResult
where
    F: FnMut(u32),
{
    let mut tokens = Vec::new();
    let mut token = first_token;

    for _ in 0..max_tokens {
        token = generate_lazy(model, state, token, config, debug);

        // Check for EOS
        if eos_token_ids.contains(&token) {
            return GenerationResult {
                tokens,
                stopped_at_eos: true,
            };
        }

        on_token(token);
        tokens.push(token);
        state.pos += 1;
    }

    GenerationResult {
        tokens,
        stopped_at_eos: false,
    }
}
