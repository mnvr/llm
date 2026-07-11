use std::collections::BTreeMap;
use std::fs;

use llm::json;

fn main() {
    let text = fs::read_to_string("models/qwen3-4b-base/model.safetensors.index.json")
        .expect("model.safetensors.index.json should exist - did the download finish?");
    let index = json::parse(&text).expect("model.safetensors.index.json should be json");
    let weight_map = index
        .get("weight_map")
        .expect("weight_map should exist")
        .as_object()
        .expect("weight_map should be an object");
    let mut names_by_shard: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    for (name, shard) in weight_map {
        let shard = shard.as_str().expect("shard should be a string");
        names_by_shard.entry(shard).or_default().push(name);
    }
    for (shard, names) in &names_by_shard {
        println!("{shard}: {} tensors", names.len());
        for name in names {
            println!("\t{name}");
        }
    }
    let (shard, names) = names_by_shard
        .iter()
        .min_by_key(|(_, names)| names.len())
        .expect("weight_map should not be empty");
    println!("smallest shard {shard} contains: {names:#?}");
}
