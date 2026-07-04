fn distinct_residues(b: usize, m: usize) -> usize {
    let mut seen = vec![false; m];
    for j in 0..m {
        seen[(b * j) % m] = true;
    }
    seen.into_iter().filter(|&s| s).count()
}

fn main() {
    for (b, m) in [(13, 101), (7, 101), (10, 100), (12, 100), (13, 100)] {
        println!("j -> {b:2}*j mod {m:3}: {:3} of {m} residues hit", distinct_residues(b, m));
    }
}
