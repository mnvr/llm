use std::time::Instant;

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

    fn matmul_ikj(&self, b: &Mat) -> Mat {
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

fn bench(name: &str, a: &Mat, b: &Mat, f: fn(&Mat, &Mat) -> Mat) -> Mat {
    let t = Instant::now();
    let c = f(a, b);
    let dt = t.elapsed().as_secs_f64();
    let flops = 2.0 * a.rows as f64 * a.cols as f64 * b.cols as f64;
    println!("{name}: {dt:5.3}s  {:5.2} GFLOPS", flops / dt / 1e9);
    c
}

fn main() {
    let a = Mat::from_fn(2, 3, |i, j| (i * 3 + j + 1) as f32);
    let b = Mat::from_fn(3, 2, |i, j| (i * 2 + j + 7) as f32);
    let c = a.matmul(&b);
    assert_eq!(c.data, vec![58.0, 64.0, 139.0, 154.0]);
    println!("hand check passed");

    let n = 1024;
    let a = Mat::from_fn(n, n, |i, j| ((i * 4 + j * 13) % 101) as f32 / 101.0);
    let b = Mat::from_fn(n, n, |i, j| ((i * 11 + j * 5) % 103) as f32 / 103.0);
    let c1 = bench("ijk", &a, &b, Mat::matmul);
    let c2 = bench("ikj", &a, &b, Mat::matmul_ikj);
    assert_eq!(c1.data, c2.data);
    println!("ijk and ikj agree bit for bit (n = {})", n);
}
