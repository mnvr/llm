use std::fs;

use llm::json::{self, Json};

fn main() {
    let text = fs::read_to_string("models/qwen3-4b-base/tokenizer.json")
        .expect("model should be downloaded");
    let json = json::parse(&text).expect("tokenizer.json should be valid json");
    for (key, value) in json.as_object().expect("top level should be an object") {
        describe(key, value);
    }
    println!();
    let model = json.get("model").expect("model should exist");
    for (key, value) in model.as_object().expect("model should be an object") {
        describe(key, value);
    }
    println!();
    let first = json
        .get("added_tokens")
        .unwrap()
        .as_array()
        .unwrap()
        .first()
        .unwrap();
    println!("added_token[0]: {first:?}");
    println!();
    let vocab = model.get("vocab").unwrap().as_object().unwrap();
    for (token, id) in &vocab[..8] {
        println!("{}: {token:?}", id.as_usize().unwrap());
    }
    println!("a -> {:?}", vocab.iter().find(|(token, _)| token == "a"));
    println!(
        "space -> {:?}",
        vocab.iter().find(|(token, _)| token == " ")
    );
    println!();
    let merges = model.get("merges").unwrap().as_array().unwrap();
    for merge in &merges[..8] {
        println!("{merge:?}");
    }

    println!("{:#x}", 'Ġ' as u32);
    println!("space encoded -> {:?}", vocab.iter().find(|(token, _)| token == "\u{120}"));
    println!("Ġ -> {:?}", vocab.iter().find(|(token, _)| token == "Ġ"));
    println!("Ā -> {:?}", vocab.iter().find(|(token, _)| token == "Ā"));
    println!("del -> {:?}", vocab.iter().find(|(token, _)| token == "\u{121}"));

    for key in ["normalizer", "pre_tokenizer", "post_processor", "decoder"] {
        println!("{key}: {:#?}", json.get(key).unwrap());
    }
}

fn describe(key: &str, value: &Json) {
    match value {
        Json::Object(entries) => println!("{key}: object, {} entries", entries.len()),
        Json::Array(items) => println!("{key}: array, {} items", items.len()),
        other => println!("{key}: {other:?}"),
    }
}
