use llm::tokenizer::Tokenizer;

fn main() {
    let tokenizer = Tokenizer::load("models/qwen3-4b-base/tokenizer.json")
        .expect("tokenizer should be downloaded");

    for text in ["  Hello", "We've 12 cats!", "def main():\n    pass\n"] {
        let ids = tokenizer.encode(text);
        println!("{text:?} -> {ids:?} -> {:?}", tokenizer.decode(&ids));
    }
}
