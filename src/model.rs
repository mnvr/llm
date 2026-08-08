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
}
