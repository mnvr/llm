use llm::json::Json;
use llm::mat::Mat;

fn main() {
    let a = Mat::from_fn(2, 3, |i, j| (i * 3 + j + 1) as f32);
    let b = Mat::from_fn(3, 2, |i, j| (i * 2 + j + 7) as f32);
    let c = a.matmul(&b);
    assert_eq!(c.data, vec![58.0, 64.0, 139.0, 154.0]);
    println!("hand check passed");

    let doc = Json::Object(vec![
        ("model_type".to_string(), Json::String("qwen3".to_string())),
        ("hidden_size".to_string(), Json::Number(2560.0)),
        ("tie_word_embeddings".to_string(), Json::Bool(true)),
        ("rope_scaling".to_string(), Json::Null),
        (
            "sizes".to_string(),
            Json::Array(vec![
                Json::Number(0.6),
                Json::Number(1.7),
                Json::Number(4.0),
            ]),
        ),
    ]);
    println!("{doc}");
    println!("size of Json = {}", std::mem::size_of::<Json>());
}
