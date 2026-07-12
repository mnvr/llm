use std::fmt;
use std::fs;

use crate::json::{self, Json};

#[derive(Debug)]
pub struct Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub eos_token_id: u32,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
    pub tie_word_embeddings: bool,
}

#[derive(Debug)]
pub struct LoadError {
    path: String,
    msg: String,
    source: Option<Box<dyn std::error::Error + 'static>>,
}

impl fmt::Display for LoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.path, self.msg)
    }
}

impl std::error::Error for LoadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.source.as_deref()
    }
}

#[derive(Debug)]
struct ParseError(String);

impl Config {
    pub fn load(path: &str) -> Result<Config, LoadError> {
        let text = fs::read_to_string(path).map_err(|e| LoadError {
            path: path.to_string(),
            msg: "could not read file".to_string(),
            source: Some(Box::new(e)),
        })?;
        let json = json::parse(&text).map_err(|e| LoadError {
            path: path.to_string(),
            msg: "invalid json".to_string(),
            source: Some(Box::new(e)),
        })?;
        Config::from_json(&json).map_err(|e| LoadError {
            path: path.to_string(),
            msg: e.0,
            source: None,
        })
    }

    fn from_json(json: &Json) -> Result<Config, ParseError> {
        let model_type = field(json, "model_type")?.as_str();
        if model_type != Some("qwen3") {
            return Err(ParseError("model_type should be qwen3".to_string()));
        }
        if !matches!(field(json, "rope_scaling")?, Json::Null) {
            return Err(ParseError("rope_scaling should be null".to_string()));
        }
        Ok(Config {
            hidden_size: int(json, "hidden_size")?,
            intermediate_size: int(json, "intermediate_size")?,
            num_hidden_layers: int(json, "num_hidden_layers")?,
            num_attention_heads: int(json, "num_attention_heads")?,
            num_key_value_heads: int(json, "num_key_value_heads")?,
            head_dim: int(json, "head_dim")?,
            vocab_size: int(json, "vocab_size")?,
            max_position_embeddings: int(json, "max_position_embeddings")?,
            eos_token_id: u32::try_from(int(json, "eos_token_id")?)
                .map_err(|_| ParseError("eos_token_id should fit in u32".to_string()))?,
            rope_theta: num(json, "rope_theta")?,
            rms_norm_eps: num(json, "rms_norm_eps")?,
            tie_word_embeddings: flag(json, "tie_word_embeddings")?,
        })
    }
}

fn field<'a>(json: &'a Json, key: &str) -> Result<&'a Json, ParseError> {
    json.get(key)
        .ok_or_else(|| ParseError(format!("missing {key}")))
}

fn num(json: &Json, key: &str) -> Result<f64, ParseError> {
    field(json, key)?
        .as_f64()
        .ok_or_else(|| ParseError(format!("{key} should be a number")))
}

fn int(json: &Json, key: &str) -> Result<usize, ParseError> {
    field(json, key)?
        .as_usize()
        .ok_or_else(|| ParseError(format!("{key} should be an integer")))
}

fn flag(json: &Json, key: &str) -> Result<bool, ParseError> {
    field(json, key)?
        .as_bool()
        .ok_or_else(|| ParseError(format!("{key} should be a bool")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json;

    #[test]
    fn from_json_qwen3_4b() {
        let text = r#"{"architectures":["Qwen3ForCausalLM"],"attention_bias":false,"attention_dropout":0.0,"bos_token_id":151643,"eos_token_id":151643,"head_dim":128,"hidden_act":"silu","hidden_size":2560,"initializer_range":0.02,"intermediate_size":9728,"max_position_embeddings":32768,"max_window_layers":36,"model_type":"qwen3","num_attention_heads":32,"num_hidden_layers":36,"num_key_value_heads":8,"rms_norm_eps":0.000001,"rope_scaling":null,"rope_theta":1000000,"sliding_window":null,"tie_word_embeddings":true,"torch_dtype":"bfloat16","transformers_version":"4.51.0","use_cache":true,"use_sliding_window":false,"vocab_size":151936}"#;
        let config = Config::from_json(&json::parse(text).unwrap()).unwrap();
        assert_eq!(config.hidden_size, 2560);
        assert_eq!(config.eos_token_id, 151643);
        assert!(config.tie_word_embeddings);
        assert_eq!(config.rms_norm_eps, 1e-6);
    }

    #[test]
    fn from_json_rejects_unknown_model() {
        let text = r#"{"model_type":"qwen2","rope_scaling":null}"#;
        let err = Config::from_json(&json::parse(text).unwrap()).unwrap_err();
        assert!(err.0.contains("model_type"));
    }
}
