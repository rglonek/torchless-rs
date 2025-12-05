use crate::model::InferenceState;
use rand::Rng;

#[cfg(test)]
mod tests;

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

/// Generate one token using the model
pub fn generate(
    model: &crate::Mistral,
    state: &mut InferenceState,
    token: u32,
    temperature: f32,
    debug: bool,
) -> u32 {
    model.forward(state, token, debug);
    sample_multinomial(state, temperature)
}
