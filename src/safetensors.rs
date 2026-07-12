use std::collections::BTreeMap;
use std::fs;

use crate::json::{self, Json};

pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

pub struct Shard {
    bytes: Vec<u8>,
    data_start: usize,
    tensors: BTreeMap<String, TensorInfo>,
}

struct TensorInfo {
    shape: Vec<usize>,
    start: usize,
    end: usize,
}

impl Shard {
    pub fn load(path: &str) -> Shard {
        let bytes = fs::read(path).unwrap_or_else(|e| panic!("{path} should be readable: {e}"));
        Shard::parse(bytes)
    }

    fn parse(bytes: Vec<u8>) -> Shard {
        let header_len =
            usize::try_from(u64::from_le_bytes(bytes[..8].try_into().unwrap())).unwrap();
        let data_start = 8 + header_len;
        let text = str::from_utf8(&bytes[8..data_start]).expect("header should be utf-8");
        let header = json::parse(text).expect("header should be json");
        let mut tensors = BTreeMap::new();
        for (name, entry) in header.as_object().expect("header should be an object") {
            if name == "__metadata__" {
                continue;
            }
            assert_eq!(
                entry.get("dtype").and_then(Json::as_str),
                Some("BF16"),
                "{name}: unsupported dtype"
            );
            let shape: Vec<usize> = entry
                .get("shape")
                .and_then(Json::as_array)
                .unwrap_or_else(|| panic!("{name}: shape should be an array"))
                .iter()
                .map(|dim| exact_usize(dim, "shape", name))
                .collect();
            let offsets = entry
                .get("data_offsets")
                .and_then(Json::as_array)
                .unwrap_or_else(|| panic!("{name}: data_offsets should be an array"));
            let [start, end] = offsets else {
                panic!("{name}: data_offsets should be a pair")
            };
            let start = exact_usize(start, "data_offsets", name);
            let end = exact_usize(end, "data_offsets", name);
            assert!(
                start <= end && end <= bytes.len() - data_start,
                "{name}: data_offsets should be in bounds"
            );
            assert_eq!(
                end - start,
                2 * shape.iter().product::<usize>(),
                "{name}: byte size should match shape"
            );
            tensors.insert(name.clone(), TensorInfo { shape, start, end });
        }
        Shard {
            bytes,
            data_start,
            tensors,
        }
    }

    pub fn tensor(&self, name: &str) -> Tensor {
        let info = self
            .tensors
            .get(name)
            .unwrap_or_else(|| panic!("{name} should be in this shard"));
        let data = self.bytes[self.data_start + info.start..self.data_start + info.end]
            .chunks_exact(2)
            .map(|pair| {
                f32::from_bits(u32::from(u16::from_le_bytes(pair.try_into().unwrap())) << 16)
            })
            .collect();
        Tensor {
            shape: info.shape.clone(),
            data,
        }
    }
}

fn exact_usize(json: &Json, what: &str, name: &str) -> usize {
    let n = json
        .as_f64()
        .unwrap_or_else(|| panic!("{name}: {what} should be numbers"));
    assert!(
        n >= 0.0 && n.fract() == 0.0,
        "{name}: {what} should be integers"
    );
    n as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_synthetic_shard() {
        let header = br#"{"t":{"dtype":"BF16","shape":[2,2],"data_offsets":[0,8]}}"#;
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header);
        bytes.extend([0x80, 0x3F, 0x00, 0x40, 0x00, 0xBF, 0x80, 0x3E]);
        let shard = Shard::parse(bytes);
        let t = shard.tensor("t");
        assert_eq!(t.shape, [2, 2]);
        assert_eq!(t.data, [1.0, 2.0, -0.5, 0.25]);
    }
}
