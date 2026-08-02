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
}
