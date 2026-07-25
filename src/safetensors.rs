use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;
use std::fs;

use crate::error::LoadError;
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

#[derive(Debug)]
enum ParseError {
    Format(&'static str),
    Utf8(std::str::Utf8Error),
    Json(json::ParseError),
    Entry(String, &'static str),
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParseError::Format(msg) => write!(f, "{msg}"),
            ParseError::Utf8(_) => write!(f, "header is not utf-8"),
            ParseError::Json(_) => write!(f, "header is not json"),
            ParseError::Entry(name, msg) => write!(f, "{name}: {msg}"),
        }
    }
}

impl Error for ParseError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Utf8(e) => Some(e),
            Self::Json(e) => Some(e),
            _ => None,
        }
    }
}

impl Shard {
    pub fn load(path: &str) -> Result<Shard, LoadError> {
        Self::load_inner(path).map_err(|source| LoadError::new(path, source))
    }

    fn load_inner(path: &str) -> Result<Self, Box<dyn Error>> {
        let bytes = fs::read(path)?;
        Ok(Self::parse(bytes)?)
    }

    fn parse(bytes: Vec<u8>) -> Result<Shard, ParseError> {
        use ParseError as E;
        let head = bytes
            .first_chunk::<8>()
            .ok_or(E::Format("file too short"))?;
        let header_len = usize::try_from(u64::from_le_bytes(*head))
            .map_err(|_| E::Format("header length does not fit usize"))?;
        if header_len > (bytes.len() - 8) {
            return Err(E::Format("header extends past end of file"));
        }
        let data_start = 8 + header_len;
        let text = str::from_utf8(&bytes[8..data_start]).map_err(E::Utf8)?;
        let header = json::parse(text).map_err(E::Json)?;
        let entries = header
            .as_object()
            .ok_or(E::Format("header should be an object"))?;
        let mut tensors = BTreeMap::new();
        for (name, entry) in entries {
            if name == "__metadata__" {
                continue;
            }
            let dtype = entry.get("dtype").and_then(Json::as_str);
            if dtype != Some("BF16") {
                return Err(E::Entry(name.clone(), "dtype should be BF16"));
            }
            let shape: Vec<usize> = entry
                .get("shape")
                .and_then(Json::as_array)
                .ok_or_else(|| E::Entry(name.clone(), "shape should be an array"))?
                .iter()
                .map(|dim| {
                    dim.as_usize()
                        .ok_or_else(|| E::Entry(name.clone(), "shape should be integers"))
                })
                .collect::<Result<Vec<usize>, ParseError>>()?;
            let offsets = entry
                .get("data_offsets")
                .and_then(Json::as_array)
                .ok_or_else(|| E::Entry(name.clone(), "data_offsets should be an array"))?;
            let [start, end] = offsets else {
                return Err(E::Entry(name.clone(), "data_offsets should be a pair"));
            };
            let start = start
                .as_usize()
                .ok_or_else(|| E::Entry(name.clone(), "start should be an integer"))?;
            let end = end
                .as_usize()
                .ok_or_else(|| E::Entry(name.clone(), "end should be an integer"))?;
            if !(start <= end && end <= bytes.len() - data_start) {
                return Err(E::Entry(name.clone(), "data_offsets should be in bounds"));
            }
            let count = shape
                .iter()
                .try_fold(1usize, |acc, &dim| acc.checked_mul(dim));
            if count.and_then(|n| n.checked_mul(2)) != Some(end - start) {
                return Err(E::Entry(name.clone(), "byte size should match shape"));
            }
            tensors.insert(name.clone(), TensorInfo { shape, start, end });
        }
        Ok(Shard {
            bytes,
            data_start,
            tensors,
        })
    }

    pub fn tensor(&self, name: &str) -> Option<Tensor> {
        let info = self.tensors.get(name)?;
        let data = self.bytes[self.data_start + info.start..self.data_start + info.end]
            .chunks_exact(2)
            .map(|pair| {
                f32::from_bits(u32::from(u16::from_le_bytes(pair.try_into().unwrap())) << 16)
            })
            .collect();
        Some(Tensor {
            shape: info.shape.clone(),
            data,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic(header: &[u8], data: &[u8]) -> Vec<u8> {
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header);
        bytes.extend_from_slice(data);
        bytes
    }

    #[test]
    fn parses_synthetic_shard() {
        let bytes = synthetic(
            br#"{"t":{"dtype":"BF16","shape":[2,2],"data_offsets":[0,8]}}"#,
            &[0x80, 0x3F, 0x00, 0x40, 0x00, 0xBF, 0x80, 0x3E],
        );
        let shard = Shard::parse(bytes).unwrap();
        let t = shard.tensor("t").unwrap();
        assert_eq!(t.shape, [2, 2]);
        assert_eq!(t.data, [1.0, 2.0, -0.5, 0.25]);
        assert!(shard.tensor("missing").is_none());
    }

    #[test]
    fn parse_rejects_malformed_shards() {
        let headers: [&[u8]; 6] = [
            br#"[]"#,
            br#"{"t":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#,
            br#"{"t":{"dtype":"BF16","shape":[2],"data_offsets":[0]}}"#,
            br#"{"t":{"dtype":"BF16","shape":[2],"data_offsets":[4,0]}}"#,
            br#"{"t":{"dtype":"BF16","shape":[2],"data_offsets":[0,999]}}"#,
            br#"{"t":{"dtype":"BF16","shape":[3],"data_offsets":[0,8]}}"#,
        ];
        for header in headers {
            let bytes = synthetic(header, &[0; 8]);
            assert!(
                Shard::parse(bytes).is_err(),
                "{}",
                str::from_utf8(header).unwrap()
            );
        }
        assert!(Shard::parse(b"ab".to_vec()).is_err());
    }
}
