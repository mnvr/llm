struct Mat {
    rows: usize,
    cols: usize,
    data: Vec<f32>,
}

impl Mat {
    fn zeros(rows: usize, cols: usize) -> Self {
        Mat { rows, cols, data: vec![0.0; rows * cols] }
    }

    fn from_fn(rows: usize, cols: usize, f: impl Fn(usize, usize) -> f32) -> Self {
        let mut m = Self::zeros(rows, cols);
        for i in 0..rows {
            for j in 0..cols {
                m.data[i * cols + j] = f(i, j);
            }
        }
        m
    }

    fn matmul(&self, b: &Mat) -> Mat {
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
}

fn main() {
    let a = Mat::from_fn(2, 3, |i, j| (i * 3 + j + 1) as f32);
    let b = Mat::from_fn(3, 2, |i, j| (i * 2 + j + 7) as f32);
    let c = a.matmul(&b);
    assert_eq!(c.data, vec![58.0, 64.0, 139.0, 154.0]);
    println!("hand check passed");
}