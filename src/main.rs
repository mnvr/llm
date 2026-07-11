use std::fs;

use llm::config::Config;
use llm::json;

fn main() {
    let text = fs::read_to_string("models/qwen3-4b-base/config.json")
        .expect("config.json should exist - did the download finish?");
    let doc = json::parse(&text).expect("config.json should parse");
    let config = Config::from_json(&doc);
    println!("{config:#?}");
}
