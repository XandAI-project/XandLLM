use candle_core::{DType, Tensor};

use crate::error::CoreResult;
use crate::model::SamplingParams;

/// Sample the next token id from logits.
///
/// ## Pipeline (non-greedy)
///
/// 1. Temperature scaling on GPU (scalar division — no data movement).
/// 2. **Single** GPU→CPU transfer (`to_vec1`).  This is the only CUDA
///    synchronization point in the non-greedy path.
/// 3. Repetition / frequency / presence penalties applied in-place on the
///    CPU vector — sparse, O(|history|) not O(vocab_size).
/// 4. Top-K mask applied on the CPU vector (partial sort, O(n) average).
/// 5. Softmax + top-P nucleus + multinomial sample — all on CPU, single pass.
///
/// The previous implementation ran two separate `to_vec1()` calls (one inside
/// `top_p_filter` and one inside `multinomial_sample`), causing two CUDA
/// synchronisation stalls, two 1 MB D2H transfers, and one 1 MB H2D mask
/// upload per generated token.  With Gemma 3's 256 k vocabulary this was a
/// dominant bottleneck.
pub fn sample_token(
    logits: &Tensor,
    params: &SamplingParams,
    token_history: &[u32],
) -> CoreResult<u32> {
    if params.greedy {
        return greedy(logits);
    }

    // ── GPU phase ─────────────────────────────────────────────────────────────
    // Temperature is a scalar division; keep it on the GPU so the model's
    // cached logits are scaled before we pull data across PCIe.
    let scaled = apply_temperature_gpu(logits, params.temperature)?;

    // Single GPU→CPU transfer — everything from here is pure CPU.
    let mut logits_vec: Vec<f32> = scaled.to_dtype(DType::F32)?.to_vec1()?;

    // ── CPU phase ─────────────────────────────────────────────────────────────
    let penalty_window: &[u32] = match params.repeat_last_n {
        Some(n) if n < token_history.len() => &token_history[token_history.len() - n..],
        _ => token_history,
    };

    // Penalties — in-place, sparse (only iterate tokens that appeared).
    if (params.repetition_penalty - 1.0).abs() > f64::EPSILON {
        apply_repetition_penalty_cpu(&mut logits_vec, penalty_window, params.repetition_penalty);
    }
    if params.frequency_penalty.abs() > f64::EPSILON || params.presence_penalty.abs() > f64::EPSILON {
        apply_frequency_presence_penalty_cpu(
            &mut logits_vec,
            penalty_window,
            params.frequency_penalty,
            params.presence_penalty,
        );
    }

    // Top-K: zero out all but the top-k logits in-place.
    if let Some(k) = params.top_k {
        top_k_filter_cpu(&mut logits_vec, k);
    }

    // Top-P + multinomial: single pass over the CPU vector.
    sample_top_p_cpu(&logits_vec, params.top_p, params.seed)
}

// ─── GPU helpers (temperature only) ──────────────────────────────────────────

/// Greedy decoding: argmax on GPU, single scalar D2H transfer.
fn greedy(logits: &Tensor) -> CoreResult<u32> {
    let id = logits.argmax(candle_core::D::Minus1)?.to_scalar::<u32>()?;
    Ok(id)
}

/// Scale logits by 1/temperature.  Returns a clone if temperature is 1.0 or
/// non-positive (treated as no scaling / greedy-like).
fn apply_temperature_gpu(logits: &Tensor, temperature: f64) -> CoreResult<Tensor> {
    if temperature <= 0.0 || (temperature - 1.0).abs() < f64::EPSILON {
        return Ok(logits.clone());
    }
    Ok((logits / temperature)?)
}

// ─── CPU in-place helpers ─────────────────────────────────────────────────────

/// Divide each token's logit that appears in `history` by `penalty`.
///
/// Operates directly on the already-transferred `Vec<f32>` — no GPU tensor
/// allocations, no H2D uploads, O(|history|) not O(vocab_size).
fn apply_repetition_penalty_cpu(logits: &mut [f32], history: &[u32], penalty: f64) {
    if penalty <= 0.0 || (penalty - 1.0).abs() < f64::EPSILON {
        return;
    }
    let p = penalty as f32;
    let vocab = logits.len();
    for &id in history {
        let idx = id as usize;
        if idx < vocab {
            // Standard repetition penalty: divide positive logits, multiply
            // negative logits, so both are pushed toward 0 (less likely).
            if logits[idx] >= 0.0 {
                logits[idx] /= p;
            } else {
                logits[idx] *= p;
            }
        }
    }
}

/// Subtract frequency and presence penalties in-place.
///
/// O(|history|) — builds counts only for tokens in history, never touches the
/// rest of the vocabulary.
fn apply_frequency_presence_penalty_cpu(
    logits: &mut [f32],
    history: &[u32],
    frequency_penalty: f64,
    presence_penalty: f64,
) {
    let vocab = logits.len();
    // Accumulate per-token counts using a small stack-friendly scan.
    // For typical histories (< 4096 tokens) this is fast with no allocation
    // beyond the HashMap itself.
    let mut counts: std::collections::HashMap<u32, usize> =
        std::collections::HashMap::with_capacity(history.len().min(512));
    for &id in history {
        *counts.entry(id).or_insert(0) += 1;
    }
    for (id, count) in counts {
        let idx = id as usize;
        if idx < vocab {
            let penalty =
                (frequency_penalty * count as f64 + presence_penalty) as f32;
            logits[idx] -= penalty;
        }
    }
}

/// Keep only the top-`k` logits; set all others to `-∞`.
///
/// Uses a partial sort (O(n) average) so we don't pay O(n log n) for sorting
/// the full vocabulary just to zero-mask most of it.
fn top_k_filter_cpu(logits: &mut Vec<f32>, k: usize) {
    if k == 0 || k >= logits.len() {
        return;
    }
    // Build indexed pairs, partial-sort to find the k-th largest threshold.
    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    indexed.select_nth_unstable_by(k, |a, b| b.1.partial_cmp(&a.1).unwrap());
    // The threshold is the value at position k (first element outside top-k).
    let threshold = indexed[k].1;
    for v in logits.iter_mut() {
        if *v < threshold {
            *v = f32::NEG_INFINITY;
        }
    }
}

/// Softmax → top-P nucleus filter → multinomial sample — entirely on CPU.
///
/// Combines what was previously `top_p_filter` (GPU softmax + D2H + H2D) and
/// `multinomial_sample` (second GPU softmax + D2H) into a single CPU pass with
/// zero GPU involvement.
fn sample_top_p_cpu(logits: &[f32], top_p: f64, seed: Option<u64>) -> CoreResult<u32> {
    let n = logits.len();

    // ── Softmax (numerically stable, CPU) ────────────────────────────────────
    let max_logit = logits
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = logits
        .iter()
        .map(|&v| (v - max_logit).exp())
        .collect();
    let sum: f32 = probs.iter().sum();
    if sum > 0.0 {
        for p in probs.iter_mut() {
            *p /= sum;
        }
    }

    // ── Top-P nucleus filter ──────────────────────────────────────────────────
    // Only applied when top_p < 1.0; otherwise sample directly from the full
    // distribution.
    if top_p < 1.0 {
        // Sort indices by descending probability to find the nucleus.
        let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
        indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let mut cumsum = 0.0f32;
        let mut nucleus_end = n;
        for (i, (_, p)) in indexed.iter().enumerate() {
            cumsum += p;
            if cumsum as f64 >= top_p {
                nucleus_end = i + 1;
                break;
            }
        }

        // Zero out probabilities outside the nucleus.
        let mut keep = vec![false; n];
        for (idx, _) in indexed.iter().take(nucleus_end) {
            keep[*idx] = true;
        }
        let mut new_sum = 0.0f32;
        for (i, p) in probs.iter_mut().enumerate() {
            if !keep[i] {
                *p = 0.0;
            } else {
                new_sum += *p;
            }
        }
        // Re-normalise the nucleus so the CDF reaches exactly 1.0.
        if new_sum > 0.0 {
            for p in probs.iter_mut() {
                *p /= new_sum;
            }
        }
    }

    // ── Multinomial sample ────────────────────────────────────────────────────
    let r: f32 = if let Some(s) = seed {
        seeded_rand_f32(s)
    } else {
        rand_f32()
    };

    let mut cumsum = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if r <= cumsum {
            return Ok(i as u32);
        }
    }
    // Fallback: rounding errors — return the last non-zero probability token.
    Ok(probs
        .iter()
        .enumerate()
        .rfind(|(_, &p)| p > 0.0)
        .map(|(i, _)| i as u32)
        .unwrap_or((n - 1) as u32))
}

// ─── RNG helpers ──────────────────────────────────────────────────────────────

/// Generate a pseudo-random float in [0, 1) seeded from wall-clock nanoseconds.
///
/// Uses a single LCG step — intentionally lightweight; the quality is
/// sufficient for sampling and avoids pulling in an external RNG dependency.
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
        // Index 4 has logit=100, everything else is 0. After softmax index 4
        // has probability ~1.0, so top_p=0.9 must select it.
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
        let logits = cpu_logits(&[0.1, 0.1, 10.0, 0.1]);
        let params = SamplingParams {
            greedy: true,
            temperature: 1.0,
            ..Default::default()
        };
        assert_eq!(sample_token(&logits, &params, &[]).unwrap(), 2);
    }

    #[test]
    fn test_repetition_penalty_cpu_reduces_seen_token() {
        // Seen token is index 0 (logit=5.0); penalty=2.0 should halve it.
        let mut logits = vec![5.0f32, 1.0, 1.0];
        apply_repetition_penalty_cpu(&mut logits, &[0u32], 2.0);
        assert!((logits[0] - 2.5).abs() < 1e-5, "Positive logit must be divided by penalty");
        assert!((logits[1] - 1.0).abs() < 1e-5, "Unseen token must be unchanged");
    }

    #[test]
    fn test_repetition_penalty_cpu_negative_logit() {
        // Negative logits should be multiplied by the penalty (pushed lower).
        let mut logits = vec![-2.0f32, 1.0];
        apply_repetition_penalty_cpu(&mut logits, &[0u32], 2.0);
        assert!((logits[0] - (-4.0)).abs() < 1e-5, "Negative logit must be multiplied by penalty");
    }

    #[test]
    fn test_top_k_filter_cpu_keeps_k_largest() {
        let mut logits = vec![1.0f32, 5.0, 3.0, 2.0, 4.0];
        top_k_filter_cpu(&mut logits, 2);
        // The two largest are indices 1 (5.0) and 4 (4.0); the rest become -∞.
        assert!(logits[1].is_finite(), "Top-1 must be kept");
        assert!(logits[4].is_finite(), "Top-2 must be kept");
        assert_eq!(logits[0], f32::NEG_INFINITY);
        assert_eq!(logits[2], f32::NEG_INFINITY);
        assert_eq!(logits[3], f32::NEG_INFINITY);
    }

    #[test]
    fn test_sample_top_p_cpu_deterministic_with_seed() {
        let logits = vec![1.0f32; 10];
        let a = sample_top_p_cpu(&logits, 1.0, Some(42)).unwrap();
        let b = sample_top_p_cpu(&logits, 1.0, Some(42)).unwrap();
        assert_eq!(a, b, "Same seed must produce same token");
    }

    #[test]
    fn test_sample_top_p_cpu_dominated_distribution() {
        // Index 7 has a massively higher logit; top_p=0.95 must select it.
        let mut logits = vec![0.0f32; 20];
        logits[7] = 50.0;
        let id = sample_top_p_cpu(&logits, 0.95, None).unwrap();
        assert_eq!(id, 7);
    }
}
