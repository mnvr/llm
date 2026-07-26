use llm::tokenizer::Tokenizer;

fn main() {
    let tokenizer = Tokenizer::load("models/qwen3-4b-base/tokenizer.json")
        .expect("tokenizer should be downloaded");
    let ids: Vec<usize> = ["Hello", ",", "Ġworld", "!"]
        .iter()
        .map(|t| tokenizer.vocab.iter().position(|v| v == t).unwrap())
        .collect();
    println!("{ids:?} -> {:?}", tokenizer.decode(&ids));

    for text in ["Hello", " world", "!", " hello", " HELLO", "Ġworld"] {
        println!("{text:?} -> {:?}", tokenizer.merge(text));
    }

    for text in ["  Hello", "We've 12 cats!", "def main():\n    pass\n"] {
        let ids = tokenizer.encode(text);
        println!("{text:?} -> {ids:?} -> {:?}", tokenizer.decode(&ids));
    }
    println!("unsplit: {:?}", tokenizer.merge("  Hello"));
}
