use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;
use std::fs;
use std::io;

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
pub enum LoadError {
    Io { path: String, source: io::Error },
    Utf8(std::str::Utf8Error),
    Json(json::ParseError),
    Format(String),
}

impl fmt::Display for LoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LoadError::Io { path, .. } => write!(f, "could not read {path}"),
            LoadError::Utf8(_) => write!(f, "header is not utf-8"),
            LoadError::Json(_) => write!(f, "header is not json"),
            LoadError::Format(msg) => write!(f, "{msg}"),
        }
    }
}

impl Error for LoadError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            LoadError::Io { source, .. } => Some(source),
            LoadError::Utf8(e) => Some(e),
            LoadError::Json(e) => Some(e),
            LoadError::Format(_) => None,
        }
    }
}

impl From<std::str::Utf8Error> for LoadError {
    fn from(e: std::str::Utf8Error) -> LoadError {
        LoadError::Utf8(e)
    }
}

impl From<json::ParseError> for LoadError {
    fn from(e: json::ParseError) -> LoadError {
        LoadError::Json(e)
    }
}

impl Shard {
    pub fn load(path: &str) -> Result<Shard, LoadError> {
        let bytes = fs::read(path).map_err(|e| LoadError::Io {
            path: path.to_string(),
            source: e,
        })?;
        Shard::parse(bytes)
    }

    fn parse(bytes: Vec<u8>) -> Result<Shard, LoadError> {
        let head = bytes
            .first_chunk::<8>()
            .ok_or_else(|| LoadError::Format("file too short".to_string()))?;
        let header_len = usize::try_from(u64::from_le_bytes(*head)).unwrap();
        ensure(header_len <= bytes.len() - 8, || {
            "header extends past end of file".to_string()
        })?;
        let data_start = 8 + header_len;
        let text = str::from_utf8(&bytes[8..data_start])?;
        let header = json::parse(text)?;
        let entries = header
            .as_object()
            .ok_or_else(|| LoadError::Format("header should be an object".to_string()))?;
        let mut tensors = BTreeMap::new();
        for (name, entry) in entries {
            if name == "__metadata__" {
                continue;
            }
            let dtype = entry.get("dtype").and_then(Json::as_str);
            ensure(dtype == Some("BF16"), || {
                format!("{name}: dtype should be BF16, not {dtype:?}")
            })?;
            let shape: Vec<usize> = entry
                .get("shape")
                .and_then(Json::as_array)
                .ok_or_else(|| LoadError::Format(format!("{name}: shape should be an array")))?
                .iter()
                .map(|dim| {
                    dim.as_usize().ok_or_else(|| {
                        LoadError::Format(format!("{name}: shape should be integers"))
                    })
                })
                .collect::<Result<Vec<usize>, LoadError>>()?;
            let offsets = entry
                .get("data_offsets")
                .and_then(Json::as_array)
                .ok_or_else(|| {
                    LoadError::Format(format!("{name}: data_offsets should be an array"))
                })?;
            let [start, end] = offsets else {
                return Err(LoadError::Format(format!(
                    "{name}: data_offsets should be a pair"
                )));
            };
            let start = start
                .as_usize()
                .ok_or_else(|| LoadError::Format(format!("{name}: start should be an integer")))?;
            let end = end
                .as_usize()
                .ok_or_else(|| LoadError::Format(format!("{name}: end should be an integer")))?;
            ensure(start <= end && end <= bytes.len() - data_start, || {
                format!("{name}: data_offsets should be in bounds")
            })?;
            let count = shape
                .iter()
                .try_fold(1usize, |acc, &dim| acc.checked_mul(dim));
            ensure(
                count.and_then(|n| n.checked_mul(2)) == Some(end - start),
                || format!("{name}: byte size should match shape"),
            )?;
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

fn ensure(cond: bool, msg: impl FnOnce() -> String) -> Result<(), LoadError> {
    if cond {
        Ok(())
    } else {
        Err(LoadError::Format(msg()))
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
