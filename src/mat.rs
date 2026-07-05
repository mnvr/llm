pub struct Mat {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}

impl Mat {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Mat {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    pub fn from_fn(rows: usize, cols: usize, f: impl Fn(usize, usize) -> f32) -> Self {
        let mut m = Self::zeros(rows, cols);
        for i in 0..rows {
            for j in 0..cols {
                m.data[i * cols + j] = f(i, j);
            }
        }
        m
    }

    pub fn matmul(&self, b: &Mat) -> Mat {
        assert_eq!(self.cols, b.rows);
        let mut c = Mat::zeros(self.rows, b.cols);
        for i in 0..self.rows {
            for j in 0..b.cols {
                let mut acc = 0.0;
                for k in 0..self.cols {
                    acc += self.data[i * self.cols + k] * b.data[k * b.cols + j];
                }
                c.data[i * b.cols + j] = acc;
            }
        }
        c
    }

    pub fn matmul_ikj(&self, b: &Mat) -> Mat {
        assert_eq!(self.cols, b.rows);
        let mut c = Mat::zeros(self.rows, b.cols);
        for i in 0..self.rows {
            for k in 0..self.cols {
                let aik = self.data[i * self.cols + k];
                for j in 0..b.cols {
                    c.data[i * b.cols + j] += aik * b.data[k * b.cols + j];
                }
            }
        }
        c
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matmul_matches_hand_computed() {
        let a = Mat::from_fn(2, 3, |i, j| (i * 3 + j + 1) as f32);
        let b = Mat::from_fn(3, 2, |i, j| (i * 2 + j + 7) as f32);
        assert_eq!(a.matmul(&b).data, [58.0, 64.0, 139.0, 154.0]);
        assert_eq!(a.matmul(&b).data, a.matmul_ikj(&b).data);
    }
}
