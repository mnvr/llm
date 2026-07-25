use std::error::Error;
use std::fmt;
use std::fs;

use crate::error::LoadError;
use crate::json::{self, Json};

pub struct Tokenizer {
    pub vocab: Vec<String>,
}

#[derive(Debug)]
struct ParseError(String);

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Error for ParseError {}

impl Tokenizer {
    pub fn load(path: &str) -> Result<Tokenizer, LoadError> {
        Self::load_inner(path).map_err(|source| LoadError::new(path, source))
    }

    pub fn decode(&self, ids: &[usize]) -> String {
        let bytes: Vec<u8> = ids
            .iter()
            .flat_map(|&id| self.vocab[id].chars())
            .map(|c| char_byte(c).unwrap())
            .collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }

    fn load_inner(path: &str) -> Result<Tokenizer, Box<dyn Error>> {
        let text = fs::read_to_string(path)?;
        let json = json::parse(&text)?;
        Ok(Self::from_json(&json)?)
    }

    fn from_json(json: &Json) -> Result<Tokenizer, ParseError> {
        let entries = json
            .get("model")
            .and_then(|model| model.get("vocab"))
            .and_then(Json::as_object)
            .ok_or_else(|| ParseError("missing model.vocab".to_string()))?;
        let mut vocab = vec![String::new(); entries.len()];
        for (token, id) in entries {
            if !token.chars().all(|c| char_byte(c).is_some()) {
                return Err(ParseError(format!("token {token:?} should be byte-level")));
            }
            let id = id
                .as_usize()
                .filter(|&id| id < vocab.len())
                .ok_or_else(|| {
                    ParseError(format!(
                        "token {token:?} should have an integer id below {}",
                        vocab.len()
                    ))
                })?;
            if !vocab[id].is_empty() {
                return Err(ParseError(format!("id {id} should be unique")));
            }
            vocab[id] = token.clone();
        }
        Ok(Tokenizer { vocab })
    }
}

pub fn byte_char(b: u8) -> char {
    let c = match b {
        0x21..=0x7e | 0xa1..=0xac | 0xae..=0xff => u32::from(b),
        0x00..=0x20 => 0x100 + u32::from(b),
        0x7f..=0xa0 => 0x121 + u32::from(b - 0x7f),
        0xad => 0x143,
    };
    char::from_u32(c).unwrap()
}

fn char_byte(c: char) -> Option<u8> {
    let c = u32::from(c);
    let b = match c {
        0x21..=0x7e | 0xa1..=0xac | 0xae..=0xff => c,
        0x100..=0x120 => c - 0x100,
        0x121..=0x142 => c - 0x121 + 0x7f,
        0x143 => 0xad,
        _ => return None,
    };
    Some(u8::try_from(b).unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_json_places_tokens_by_id() {
        let text = r#"{"model":{"vocab":{"b":1,"a":0}}}"#;
        let tokenizer = Tokenizer::from_json(&json::parse(text).unwrap()).unwrap();
        assert_eq!(tokenizer.vocab, ["a", "b"]);
    }

    #[test]
    fn from_json_rejects_malformed_vocabs() {
        let texts = [
            r#"{}"#,
            r#"{"model":{}}"#,
            r#"{"model":{"vocab":[]}}"#,
            r#"{"model":{"vocab":{"a":true}}}"#,
            r#"{"model":{"vocab":{"a":0,"b":2}}}"#,
            r#"{"model":{"vocab":{"a":0,"b":0}}}"#,
            r#"{"model":{"vocab":{"€":0}}}"#,
        ];
        for text in texts {
            assert!(
                Tokenizer::from_json(&json::parse(text).unwrap()).is_err(),
                "{text}"
            )
        }
    }

    #[test]
    fn decode_reverses_spellings() {
        let text = r#"{"model":{"vocab":{"Ġhello":0,"Ã©":1,"!":2}}}"#;
        let tokenizer = Tokenizer::from_json(&json::parse(text).unwrap()).unwrap();
        assert_eq!(tokenizer.decode(&[0, 1, 2]), " helloé!");
    }

    #[test]
    fn decode_replaces_invalid_utf8() {
        let text = r#"{"model":{"vocab":{"Ã":0}}}"#;
        let tokenizer = Tokenizer::from_json(&json::parse(text).unwrap()).unwrap();
        assert_eq!(tokenizer.decode(&[0]), "\u{fffd}");
    }

    #[test]
    fn byte_char_matches_known_spellings() {
        assert_eq!(byte_char(0x21), '!');
        assert_eq!(byte_char(0x61), 'a');
        assert_eq!(byte_char(0xff), 'ÿ');
        assert_eq!(byte_char(0x20), 'Ġ');
        assert_eq!(byte_char(0x00), 'Ā');
        assert_eq!(byte_char(0x7f), 'ġ');
        assert_eq!(byte_char(0xa0), 'ł');
        assert_eq!(byte_char(0xad), 'Ń');
        assert_eq!(byte_char(0xb7), '·');
    }

    #[test]
    fn char_byte_inverts_byte_char() {
        for b in 0..=u8::MAX {
            assert_eq!(char_byte(byte_char(b)), Some(b));
        }
        assert_eq!(char_byte('€'), None);
    }
}
