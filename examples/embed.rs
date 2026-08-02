use std::fs;

use llm::json;
use llm::safetensors::Shard;
use llm::tokenizer::Tokenizer;

fn main() {
    let text = fs::read_to_string("models/qwen3-4b-base/model.safetensors.index.json").unwrap();
    let index = json::parse(&text).unwrap();
    let file = index
        .get("weight_map")
        .and_then(|map| map.get("model.embed_tokens.weight"))
        .and_then(|file| file.as_str())
        .unwrap();
    println!("{file}");
    let shard = Shard::load(&format!("models/qwen3-4b-base/{file}")).unwrap();
    let embed = shard.tensor("model.embed_tokens.weight").unwrap();
    println!("{:?}", embed.shape);
    let tokenizer = Tokenizer::load("models/qwen3-4b-base/tokenizer.json").unwrap();
    for id in tokenizer.encode("Hello world") {
        let dim = embed.shape[1];
        println!("{id}: {:?}", &embed.data[id * dim..id * dim + 4]);
    }
}
