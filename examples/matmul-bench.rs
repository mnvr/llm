use std::time::Instant;

use llm::mat::Mat;

fn bench(name: &str, a: &Mat, b: &Mat, f: fn(&Mat, &Mat) -> Mat) -> Mat {
    let t = Instant::now();
    let c = f(a, b);
    let dt = t.elapsed().as_secs_f64();
    let flops = 2.0 * a.rows as f64 * a.cols as f64 * b.cols as f64;
    println!("{name}: {dt:5.3}s  {:5.2} GFLOPS", flops / dt / 1e9);
    c
}

fn main() {
    let n = 1024;
    let a = Mat::from_fn(n, n, |i, j| ((i * 7 + j * 13) % 101) as f32 / 101.0);
    let b = Mat::from_fn(n, n, |i, j| ((i * 11 + j * 5) % 103) as f32 / 103.0);
    let c1 = bench("ijk", &a, &b, Mat::matmul);
    let c2 = bench("ikj", &a, &b, Mat::matmul_ikj);
    assert_eq!(c1.data, c2.data);
    println!("ijk and ikj agree bit for bit (n = {n})");
}
