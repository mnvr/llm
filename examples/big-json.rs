use std::fs;
use std::time::Instant;

use llm::json;

fn main() {
    let t = Instant::now();
    let text = fs::read_to_string("models/qwen3-4b-base/tokenizer.json")
        .expect("tokenizer.json should exist - did the download finish?");
    let dt_read = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let doc = json::parse(&text).expect("tokenizer.json should parse");
    let dt_parse = t.elapsed().as_secs_f64();
    let t = Instant::now();
    drop(doc);
    let dt_drop = t.elapsed().as_secs_f64();
    println!(
        "tokenizer.json ({} bytes): read {dt_read:.3}s parse {dt_parse:.3}s drop {dt_drop:.3}s",
        text.len()
    );
}
