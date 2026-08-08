use std::fs;

use llm::tokenizer::Tokenizer;

fn main() {
    let tokenizer =
        Tokenizer::load("models/qwen3-4b-base/tokenizer.json").expect("model should be downloaded");
    let corpus = fs::read_to_string("reference/corpus.txt").expect("corpus should exist");
    for line in corpus.split_inclusive('\n') {
        println!("{:?}", tokenizer.encode(line));
    }
}
