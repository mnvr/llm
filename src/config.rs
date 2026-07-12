use std::fmt;

use crate::json::Json;

#[derive(Debug)]
pub struct Error(String);

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "config: {}", self.0)
    }
}

impl std::error::Error for Error {}

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

impl Config {
    pub fn from_json(json: &Json) -> Result<Config, Error> {
        let model_type = field(json, "model_type")?.as_str();
        if model_type != Some("qwen3") {
            return Err(Error(format!(
                "model_type should be qwen3, not {model_type:?}"
            )));
        }
        if !matches!(field(json, "rope_scaling")?, Json::Null) {
            return Err(Error("rope_scaling should be null".to_string()));
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
                .map_err(|_| Error("eos_token_id should fit in u32".to_string()))?,
            rope_theta: num(json, "rope_theta")?,
            rms_norm_eps: num(json, "rms_norm_eps")?,
            tie_word_embeddings: flag(json, "tie_word_embeddings")?,
        })
    }
}

fn field<'a>(json: &'a Json, key: &str) -> Result<&'a Json, Error> {
    json.get(key).ok_or_else(|| Error(format!("missing {key}")))
}

fn num(json: &Json, key: &str) -> Result<f64, Error> {
    field(json, key)?
        .as_f64()
        .ok_or_else(|| Error(format!("{key} should be a number")))
}

fn int(json: &Json, key: &str) -> Result<usize, Error> {
    field(json, key)?
        .as_usize()
        .ok_or_else(|| Error(format!("config: {key} should be an integer")))
}

fn flag(json: &Json, key: &str) -> Result<bool, Error> {
    field(json, key)?
        .as_bool()
        .ok_or_else(|| Error(format!("config: {key} should be a bool")))
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
    fn from_json_reject_unknown_model() {
        let text = r#"{"model_type":"qwen2","rope_scaling":null}"#;
        let err = Config::from_json(&json::parse(text).unwrap()).unwrap_err();
        assert!(err.0.contains("model_type"));
    }
}
