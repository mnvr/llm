pub fn rms_norm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let mean = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let scale = 1.0 / (mean + eps).sqrt();
    x.iter().zip(weight).map(|(v, w)| v * scale * w).collect()
}

pub fn matvec(w: &[f32], x: &[f32]) -> Vec<f32> {
    assert_eq!(w.len() % x.len(), 0);
    w.chunks_exact(x.len())
        .map(|row| row.iter().zip(x).map(|(w, x)| w * x).sum())
        .collect()
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

pub fn mlp(x: &[f32], gate: &[f32], up: &[f32], down: &[f32]) -> Vec<f32> {
    let g = matvec(gate, x);
    let u = matvec(up, x);
    let hidden: Vec<f32> = g.iter().zip(&u).map(|(&g, &u)| silu(g) * u).collect();
    matvec(down, &hidden)
}

pub fn softmax(x: &[f32]) -> Vec<f32> {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<_> = x.iter().map(|v| (v - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|v| v / sum).collect()
}

pub fn attend(q: &[f32], keys: &[f32], values: &[f32]) -> Vec<f32> {
    let scale = 1.0 / (q.len() as f32).sqrt();
    let scores: Vec<f32> = matvec(keys, q).iter().map(|s| s * scale).collect();
    let weights = softmax(&scores);
    let mut out = vec![0.0; q.len()];
    for (&w, row) in weights.iter().zip(values.chunks_exact(q.len())) {
        for (o, &v) in out.iter_mut().zip(row) {
            *o += w * v;
        }
    }
    out
}

pub fn rope(x: &[f32], pos: usize, theta: f32) -> Vec<f32> {
    let half = x.len() / 2;
    let mut out = vec![0.0; x.len()];
    for p in 0..half {
        let angle = pos as f32 * theta.powf(-2.0 * p as f32 / x.len() as f32);
        let (sin, cos) = angle.sin_cos();
        out[p] = x[p] * cos - x[p + half] * sin;
        out[p + half] = x[p] * sin + x[p + half] * cos;
    }
    out
}

#[derive(Clone, Default)]
pub struct HeadCache {
    pub keys: Vec<f32>,
    pub values: Vec<f32>,
}

pub fn attention(
    x: &[f32],
    q_proj: &[f32],
    k_proj: &[f32],
    v_proj: &[f32],
    o_proj: &[f32],
    q_norm: &[f32],
    k_norm: &[f32],
    cache: &mut [HeadCache],
    pos: usize,
    theta: f32,
    eps: f32,
) -> Vec<f32> {
    let dim = q_norm.len();
    let q_all = matvec(q_proj, x);
    let k_all = matvec(k_proj, x);
    let v_all = matvec(v_proj, x);
    for (head, (k, v)) in cache
        .iter_mut()
        .zip(k_all.chunks_exact(dim).zip(v_all.chunks_exact(dim)))
    {
        head.keys
            .extend(rope(&rms_norm(k, k_norm, eps), pos, theta));
        head.values.extend_from_slice(v);
    }
    let group = q_all.len() / k_all.len();
    let mut merged = Vec::with_capacity(q_all.len());
    for (i, chunk) in q_all.chunks_exact(dim).enumerate() {
        let q = rope(&rms_norm(chunk, q_norm, eps), pos, theta);
        let head = &cache[i / group];
        merged.extend(attend(&q, &head.keys, &head.values));
    }
    matvec(o_proj, &merged)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rms_norm_scales_to_unit_rms() {
        let x = [3.0, 3.0, -3.0, 3.0];
        let w = [1.0, 1.0, 1.0, 2.0];
        assert_eq!(rms_norm(&x, &w, 0.0), [1.0, 1.0, -1.0, 2.0]);
    }

    #[test]
    fn rms_norm_adds_eps_inside_the_root() {
        assert_eq!(rms_norm(&[1.0; 4], &[1.0; 4], 3.0), [0.5; 4]);
    }

    #[test]
    fn matvec_dots_each_row_with_x() {
        let w = [1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        assert_eq!(matvec(&w, &[3.0, 4.0]), [3.0, 4.0, 7.0]);
    }

    #[test]
    fn silu_passes_large_positive_and_blocks_large_negative() {
        assert_eq!(silu(0.0), 0.0);
        assert_eq!(silu(100.0), 100.0);
        assert_eq!(silu(-100.0), 0.0);
    }

    #[test]
    fn mlp_scales_each_up_channel_by_its_gate_then_projects_down() {
        let x = [1.0, 2.0];
        let gate = [2.0, -1.0, 100.0, 0.0, -100.0, 0.0];
        let up = [5.0, 5.0, 1.0, 1.0, 7.0, 0.0];
        let down = [1.0, 1.0, 1.0, 0.0, 0.5, 10.0];
        assert_eq!(mlp(&x, &gate, &up, &down), [300.0, 150.0]);
    }

    #[test]
    fn softmax_shares_weight_equally_between_equal_scores() {
        assert_eq!(softmax(&[3.0; 4]), [0.25; 4]);
    }

    #[test]
    fn softmax_ignores_a_common_shift() {
        assert_eq!(softmax(&[1.0, 2.0]), softmax(&[1001.0, 1002.0]));
    }

    #[test]
    fn softmax_gives_all_weight_to_a_runaway_winner() {
        assert_eq!(softmax(&[-1000.0, 0.0]), [0.0, 1.0]);
    }

    #[test]
    fn attend_with_one_position_returns_its_value() {
        let q = [1.0, 0.0, 0.0, 0.0];
        assert_eq!(
            attend(&q, &[7.0; 4], &[1.0, 2.0, 3.0, 4.0]),
            [1.0, 2.0, 3.0, 4.0]
        );
    }

    #[test]
    fn attend_averages_the_values_of_equally_matching_keys() {
        let q = [1.0, 1.0, 0.0, 0.0];
        let keys = [2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0];
        let values = [4.0, 0.0, 2.0, 0.0, 0.0, 4.0, 2.0, 8.0];
        assert_eq!(attend(&q, &keys, &values), [2.0, 2.0, 2.0, 4.0]);
    }

    #[test]
    fn attend_picks_the_value_of_a_dominant_key() {
        let q = [1.0, 0.0, 0.0, 0.0];
        let keys = [0.0, 9.0, 9.0, 9.0, 4000.0, 0.0, 0.0, 0.0];
        let values = [9.0, 9.0, 9.0, 9.0, 5.0, 6.0, 7.0, 8.0];
        assert_eq!(attend(&q, &keys, &values), [5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn rope_at_position_zero_changes_nothing() {
        let x = [0.5, -1.0, 2.0, 3.0];
        assert_eq!(rope(&x, 0, 1000000.0), x);
    }

    #[test]
    fn rope_pairs_dimension_p_with_p_plus_half() {
        assert_eq!(
            rope(&[1.0, 0.0, 0.0, 0.0], 1, 1.0),
            [1f32.cos(), 0.0, 1f32.sin(), 0.0]
        );
        assert_eq!(
            rope(&[0.0, 0.0, 1.0, 0.0], 1, 1.0),
            [-(1f32.sin()), 0.0, 1f32.cos(), 0.0]
        );
    }

    #[test]
    fn rope_rotates_later_planes_slower() {
        let out = rope(&[1.0, 1.0, 0.0, 0.0], 1, 1000000.0);
        assert!(out[1] > 0.99 && out[0] < 0.55);
    }

    #[test]
    fn attention_first_position_routes_values_by_group() {
        let x = [1.0, 0.0];
        let q_proj = [
            8.0, 0.0, 8.0, 0.0, 8.0, 0.0, 8.0, 0.0, 8.0, 0.0, 8.0, 0.0, 8.0, 0.0, 8.0, 0.0,
        ];
        let k_proj = [8.0, 0.0, 8.0, 0.0, 8.0, 0.0, 8.0, 0.0];
        let v_proj = [1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0];
        let o_proj = [
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
        ];
        let norm = [1.0, 1.0];
        let mut cache = vec![HeadCache::default(); 2];
        let out = attention(
            &x, &q_proj, &k_proj, &v_proj, &o_proj, &norm, &norm, &mut cache, 0, 1000000.0, 0.0,
        );
        assert_eq!(out, [1.0, 3.0]);
    }

    #[test]
    fn attention_mixes_two_cached_positions_equally() {
        let q_proj = [8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0];
        let k_proj = [8.0, -8000.0, 8.0, 8000.0];
        let v_proj = [1.0, 3.0, 2.0, 4.0];
        let o_proj = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        let q_norm = [0.0, 1.0];
        let k_norm = [3.0, 1.0];
        let mut cache = vec![HeadCache::default()];
        let first = attention(
            &[1.0, 0.0],
            &q_proj,
            &k_proj,
            &v_proj,
            &o_proj,
            &q_norm,
            &k_norm,
            &mut cache,
            0,
            1000000.0,
            0.0,
        );
        assert_eq!(first, [1.0, 2.0]);
        let second = attention(
            &[0.0, 1.0],
            &q_proj,
            &k_proj,
            &v_proj,
            &o_proj,
            &q_norm,
            &k_norm,
            &mut cache,
            0,
            1000000.0,
            0.0,
        );
        assert_eq!(second, [2.0, 3.0]);
    }

    #[test]
    fn attention_caches_rotated_keys_and_raw_values() {
        let q_proj = [1.0, 0.0, 1.0, 0.0];
        let k_proj = [8.0, 0.0, 8.0, 0.0];
        let v_proj = [5.0, 0.0, 6.0, 0.0];
        let o_proj = [1.0, 0.0, 0.0, 1.0];
        let q_norm = [9.0, 9.0];
        let k_norm = [1.0, 1.0];
        let mut cache = vec![HeadCache::default()];
        let out = attention(
            &[1.0, 0.0],
            &q_proj,
            &k_proj,
            &v_proj,
            &o_proj,
            &q_norm,
            &k_norm,
            &mut cache,
            1,
            1000000.0,
            0.0,
        );
        assert_eq!(out, [5.0, 6.0]);
        assert_eq!(cache[0].keys, rope(&[1.0, 1.0], 1, 1000000.0));
        assert_eq!(cache[0].values, [5.0, 6.0]);
    }
}
