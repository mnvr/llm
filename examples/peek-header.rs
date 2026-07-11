use std::fs;

use llm::json;

fn main() {
    let bytes = fs::read("models/qwen3-4b-base/model-00003-of-00003.safetensors")
        .expect("shard should exist - did the download finish?");

    let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap());
    let header_len = usize::try_from(header_len).unwrap();
    println!("header length: {header_len} bytes");

    let header = str::from_utf8(&bytes[8..8 + header_len]).unwrap();
    println!("{header}");

    let header = json::parse(header).expect("header should be json");
    let data = &bytes[8 + header_len..];

    let offsets = header
        .get("model.norm.weight")
        .expect("model.norm.weight should be in this shard")
        .get("data_offsets")
        .expect("tensor should have data_offsets")
        .as_array()
        .expect("data_offsets should be an array");
    let start = offsets[0].as_f64().unwrap() as usize;
    let end = offsets[1].as_f64().unwrap() as usize;
    let weight = &data[start..end];

    for i in 0..8 {
        let bits = u16::from_le_bytes(weight[2 * i..2 * i + 2].try_into().unwrap());
        println!("{}", f32::from_bits(u32::from(bits) << 16));
    }
}
