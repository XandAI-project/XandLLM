use candle_core::{DType, Tensor};
use std::collections::HashMap;

use crate::error::CoreResult;
use crate::model::SamplingParams;

/// Sample the next token id from logits.
///
/// Applies various penalties and filters in the following order:
/// 1. Repetition penalty (based on token history)
/// 2. Frequency/presence penalties (OpenAI-style)
/// 3. Top-K filtering
/// 4. Temperature scaling
/// 5. Top-P (nucleus) filtering
/// 6. Multinomial sampling (with optional seeded RNG)
///
/// `token_history` should include all tokens generated so far (including prompt tokens).
pub fn sample_token(
    logits: &Tensor,
    params: &SamplingParams,
    token_history: &[u32],
) -> CoreResult<u32> {
    if params.greedy {
        return greedy(logits);
    }

    // Apply the repeat_last_n window — cap how far back penalty scans go.
    // This bounds penalty computation to O(window) instead of O(T) and
    // prevents unbounded token_history from consuming growing memory,
    // matching llama.cpp's `repeat_last_n` semantics.
    let penalty_window: &[u32] = match params.repeat_last_n {
        Some(n) if n < token_history.len() => {
            &token_history[token_history.len() - n..]
        }
        _ => token_history,
    };

    let mut logits = logits.clone();

    // Apply repetition penalty
    if (params.repetition_penalty - 1.0).abs() > f64::EPSILON {
        logits = apply_repetition_penalty(&logits, penalty_window, params.repetition_penalty)?;
    }

    // Apply frequency and presence penalties
    if params.frequency_penalty.abs() > f64::EPSILON || params.presence_penalty.abs() > f64::EPSILON {
        logits = apply_frequency_presence_penalty(
            &logits,
            penalty_window,
            params.frequency_penalty,
            params.presence_penalty,
        )?;
    }

    // Apply top-k filter
    if let Some(k) = params.top_k {
        logits = top_k_filter(&logits, k)?;
    }

    // Apply temperature
    logits = apply_temperature(&logits, params.temperature)?;

    // Apply top-p filter
    logits = top_p_filter(&logits, params.top_p)?;

    // Sample with optional seed
    multinomial_sample(&logits, params.seed)
}

/// Greedy decoding: return the token with the highest logit.
fn greedy(logits: &Tensor) -> CoreResult<u32> {
    let id = logits
        .argmax(candle_core::D::Minus1)?
        .to_scalar::<u32>()?;
    Ok(id)
}

/// Divide logits by temperature. A temperature of 0 is treated as greedy.
fn apply_temperature(logits: &Tensor, temperature: f64) -> CoreResult<Tensor> {
    if temperature <= 0.0 || temperature == 1.0 {
        return Ok(logits.clone());
    }
    Ok((logits / temperature)?)
}

/// Apply repetition penalty to discourage repeating tokens.
///
/// For each token that appears in `token_history`, divide its logit by `penalty`.
/// A penalty > 1.0 makes repeated tokens less likely.
/// 
/// GPU-optimized: builds mask on CPU, performs division on GPU (no GPU→CPU transfer).
fn apply_repetition_penalty(
    logits: &Tensor,
    token_history: &[u32],
    penalty: f64,
) -> CoreResult<Tensor> {
    if penalty <= 0.0 || (penalty - 1.0).abs() < f64::EPSILON {
        return Ok(logits.clone());
    }

    let vocab_size = logits.dims1()?;
    let mut penalty_mask = vec![1.0f32; vocab_size];
    
    for &token_id in token_history {
        let idx = token_id as usize;
        if idx < vocab_size {
            penalty_mask[idx] = penalty as f32;
        }
    }

    let mask = Tensor::from_vec(penalty_mask, vocab_size, logits.device())?;
    Ok((logits / mask)?)
}

/// Apply OpenAI-style frequency and presence penalties.
///
/// - Frequency penalty: scales with how many times a token has appeared
/// - Presence penalty: binary (0 or 1 appearance)
/// 
/// GPU-optimized: builds penalty vector on CPU, performs subtraction on GPU.
fn apply_frequency_presence_penalty(
    logits: &Tensor,
    token_history: &[u32],
    frequency_penalty: f64,
    presence_penalty: f64,
) -> CoreResult<Tensor> {
    let vocab_size = logits.dims1()?;
    let mut penalty_vec = vec![0.0f32; vocab_size];
    
    let mut token_counts: HashMap<u32, usize> = HashMap::new();
    for &token_id in token_history {
        *token_counts.entry(token_id).or_insert(0) += 1;
    }

    for (&token_id, &count) in &token_counts {
        let idx = token_id as usize;
        if idx < vocab_size {
            let freq = frequency_penalty * count as f64;
            let pres = if count > 0 { presence_penalty } else { 0.0 };
            penalty_vec[idx] = (freq + pres) as f32;
        }
    }

    let penalty_tensor = Tensor::from_vec(penalty_vec, vocab_size, logits.device())?;
    Ok((logits - penalty_tensor)?)
}

/// Keep only the top K logits, zero out the rest.
/// 
/// GPU-optimized: single transfer for sorting, uses partial sort for O(n) instead of O(n log n).
fn top_k_filter(logits: &Tensor, k: usize) -> CoreResult<Tensor> {
    if k == 0 {
        return Ok(logits.clone());
    }

    let vocab_size = logits.dims1()?;
    
    if k >= vocab_size {
        return Ok(logits.clone());
    }

    let logits_vec: Vec<f32> = logits.to_dtype(DType::F32)?.to_vec1()?;
    let mut indexed: Vec<(usize, f32)> = logits_vec.iter().copied().enumerate().collect();
    
    // Partial sort: O(n) average case instead of O(n log n) full sort
    indexed.select_nth_unstable_by(k, |a, b| b.1.partial_cmp(&a.1).unwrap());

    let mut mask = vec![f32::NEG_INFINITY; vocab_size];
    for (idx, _) in indexed.iter().take(k) {
        mask[*idx] = 0.0;
    }

    let mask_tensor = Tensor::from_vec(mask, vocab_size, logits.device())?;
    Ok((logits + mask_tensor)?)
}

/// Zero out logits outside the top-p nucleus.
/// 
/// Optimized: single GPU→CPU transfer (for sorting/cumsum), mask creation on GPU.
fn top_p_filter(logits: &Tensor, top_p: f64) -> CoreResult<Tensor> {
    if top_p >= 1.0 {
        return Ok(logits.clone());
    }

    // Softmax on GPU, then single transfer for sorting
    let probs = candle_nn::ops::softmax(logits, candle_core::D::Minus1)?;
    let probs_vec: Vec<f32> = probs.to_dtype(DType::F32)?.to_vec1()?;

    let vocab_size = probs_vec.len();

    // Sort indices by descending probability
    let mut indexed: Vec<(usize, f32)> = probs_vec.iter().copied().enumerate().collect();
    indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Find the cutoff index
    let mut cumsum = 0.0f32;
    let mut cutoff = vocab_size;
    for (i, (_, p)) in indexed.iter().enumerate() {
        cumsum += p;
        if cumsum as f64 >= top_p {
            cutoff = i + 1;
            break;
        }
    }

    // Build mask: keep only the top-p tokens
    let mut mask = vec![f32::NEG_INFINITY; vocab_size];
    for (idx, _) in indexed.iter().take(cutoff) {
        mask[*idx] = 0.0;
    }

    let mask_tensor = Tensor::from_vec(mask, vocab_size, logits.device())?;
    Ok((logits + mask_tensor)?)
}

/// Draw a sample from the (possibly filtered) logit distribution.
/// 
/// This is the only mandatory GPU→CPU transfer in the sampling pipeline
/// (random number comparison must happen on CPU).
fn multinomial_sample(logits: &Tensor, seed: Option<u64>) -> CoreResult<u32> {
    let probs = candle_nn::ops::softmax(logits, candle_core::D::Minus1)?;
    let probs_vec: Vec<f32> = probs.to_dtype(DType::F32)?.to_vec1()?;

    let sample: f32 = if let Some(s) = seed {
        seeded_rand_f32(s)
    } else {
        rand_f32()
    };

    let mut cumsum = 0.0f32;
    for (i, &p) in probs_vec.iter().enumerate() {
        cumsum += p;
        if sample <= cumsum {
            return Ok(i as u32);
        }
    }
    // Fallback to last token (rounding errors)
    Ok((probs_vec.len() - 1) as u32)
}

/// Simple LCG-based pseudo-random float in [0, 1).
///
/// Using a lightweight inline RNG avoids pulling in an extra dependency.
fn rand_f32() -> f32 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let seed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(12345);
    let x = seed.wrapping_mul(1664525).wrapping_add(1013904223);
    (x as f32) / (u32::MAX as f32)
}

/// Seeded pseudo-random float in [0, 1).
fn seeded_rand_f32(seed: u64) -> f32 {
    let x = (seed as u32).wrapping_mul(1664525).wrapping_add(1013904223);
    (x as f32) / (u32::MAX as f32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};
    use crate::model::SamplingParams;

    fn cpu_logits(values: &[f32]) -> Tensor {
        Tensor::from_vec(values.to_vec(), values.len(), &Device::Cpu).unwrap()
    }

    #[test]
    fn test_greedy_picks_argmax() {
        // Index 3 has the highest value — greedy must pick it
        let logits = cpu_logits(&[0.1, 0.2, 0.3, 5.0, 0.1]);
        let params = SamplingParams { greedy: true, ..Default::default() };
        let id = sample_token(&logits, &params, &[]).unwrap();
        assert_eq!(id, 3, "Greedy should pick the argmax token");
    }

    #[test]
    fn test_greedy_first_index_max() {
        let logits = cpu_logits(&[10.0, 1.0, 1.0, 1.0]);
        let params = SamplingParams { greedy: true, ..Default::default() };
        assert_eq!(sample_token(&logits, &params, &[]).unwrap(), 0);
    }

    #[test]
    fn test_top_p_full_does_not_panic() {
        // top_p = 1.0 means no filtering; sampling should succeed on uniform logits
        let logits = cpu_logits(&[1.0f32; 20]);
        let params = SamplingParams {
            greedy: false,
            top_p: 1.0,
            temperature: 1.0,
            ..Default::default()
        };
        let id = sample_token(&logits, &params, &[]).unwrap();
        assert!((id as usize) < 20);
    }

    #[test]
    fn test_top_p_narrow_selects_from_top() {
        // Only indices 4 (logit=100) should survive top_p=0.9
        let mut values = vec![0.0f32; 10];
        values[4] = 100.0;
        let logits = cpu_logits(&values);
        let params = SamplingParams {
            greedy: false,
            top_p: 0.9,
            temperature: 1.0,
            ..Default::default()
        };
        let id = sample_token(&logits, &params, &[]).unwrap();
        assert_eq!(id, 4, "With one dominant logit, top-p should select it");
    }

    #[test]
    fn test_temperature_passthrough_at_one() {
        // temperature == 1.0 is a passthrough; greedy result must stay the same
        let logits = cpu_logits(&[0.1, 0.1, 10.0, 0.1]);
        let params = SamplingParams {
            greedy: true,
            temperature: 1.0,
            ..Default::default()
        };
        assert_eq!(sample_token(&logits, &params, &[]).unwrap(), 2);
    }
}
