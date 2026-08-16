use llm::config::Config;
use llm::model::{HeadCache, Model};
use llm::tokenizer::Tokenizer;

fn main() {
    let dir = "models/qwen3-4b-base";
    let config = Config::load(&format!("{dir}/config.json")).unwrap();
    let tokenizer = Tokenizer::load(&format!("{dir}/tokenizer.json")).unwrap();
    let model = Model::load(dir, &config).unwrap();
    let mut cache =
        vec![vec![HeadCache::default(); config.num_key_value_heads]; config.num_hidden_layers];
    let ids = tokenizer.encode("The capital of France is");
    let (&last, rest) = ids.split_last().unwrap();
    for (pos, &id) in rest.iter().enumerate() {
        model.forward(id as u32, &mut cache, pos);
    }
    let logits = model.forward(last as u32, &mut cache, rest.len());
    let (best, score) = logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .unwrap();
    println!("{best} {score} {:?}", tokenizer.decode(&[best]));
}
