use std::collections::HashMap;
use std::error::Error;
use std::fmt;
use std::fs;

use crate::error::LoadError;
use crate::json::{self, Json};

pub struct Tokenizer {
    vocab: Vec<String>,
    ids: HashMap<String, usize>,
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

    pub fn encode(&self, text: &str) -> Vec<usize> {
        split(nfc(text))
            .iter()
            .flat_map(|chunk| self.merge(chunk))
            .collect()
    }

    fn merge(&self, chunk: &str) -> Vec<usize> {
        let mut parts: Vec<_> = chunk.bytes().map(|b| byte_char(b).to_string()).collect();
        loop {
            let best = parts
                .windows(2)
                .enumerate()
                .filter_map(|(i, pair)| {
                    let rank = self.ids.get(&format!("{}{}", pair[0], pair[1]))?;
                    Some((*rank, i))
                })
                .min();
            let Some((_, i)) = best else { break };
            let right = parts.remove(i + 1);
            parts[i].push_str(&right);
        }
        parts.iter().map(|part| self.ids[part]).collect()
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
        let ids = vocab
            .iter()
            .enumerate()
            .map(|(id, token)| (token.clone(), id))
            .collect();
        Ok(Tokenizer { vocab, ids })
    }
}

fn byte_char(b: u8) -> char {
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

fn nfc(text: &str) -> &str {
    // TODO: NFC
    text
}

fn split(text: &str) -> Vec<&str> {
    let chars: Vec<_> = text.char_indices().collect();
    let mut chunks = Vec::new();
    let mut start = 0;
    while start < chars.len() {
        let rest = &chars[start..];
        let len = contraction(rest)
            .or_else(|| word(rest))
            .or_else(|| number(rest))
            .or_else(|| punctuation(rest))
            .or_else(|| newlines(rest))
            .or_else(|| space(rest))
            .or_else(|| whitespace(rest))
            .unwrap();
        let end = start + len;
        let from = chars[start].0;
        let to = chars.get(end).map_or(text.len(), |&(i, _)| i);
        chunks.push(&text[from..to]);
        start = end;
    }
    chunks
}

fn is_letter(c: char) -> bool {
    // TODO: Rough equivalent of \p{L}
    c.is_alphabetic() && !c.is_numeric()
}

/// (?i:'s|'t|'re|'ve|'m|'ll|'d)
fn contraction(s: &[(usize, char)]) -> Option<usize> {
    if s.first()?.1 != '\'' {
        return None;
    }
    let c = |i: usize| s.get(i).map(|(_, c)| c.to_ascii_lowercase());
    match (c(1)?, c(2)) {
        ('s' | 't' | 'm' | 'd', _) => Some(2),
        ('r', Some('e')) | ('v', Some('e')) | ('l', Some('l')) => Some(3),
        _ => None,
    }
}

/// [^\r\n\p{L}\p{N}]?\p{L}+
fn word(s: &[(usize, char)]) -> Option<usize> {
    let c = s.first()?.1;
    let prefix = usize::from(!(c == '\r' || c == '\n' || is_letter(c) || c.is_numeric()));
    let letters = s[prefix..]
        .iter()
        .take_while(|&&(_, c)| is_letter(c))
        .count();
    (letters > 0).then_some(prefix + letters)
}

/// \p{N}
fn number(s: &[(usize, char)]) -> Option<usize> {
    s.first().filter(|(_, c)| c.is_numeric()).map(|_| 1)
}

///  ?[^\s\p{L}\p{N}]+[\r\n]*
fn punctuation(s: &[(usize, char)]) -> Option<usize> {
    let space = usize::from(s.first()?.1 == ' ');
    let punc = s[space..]
        .iter()
        .take_while(|&&(_, c)| !(c.is_whitespace() || is_letter(c) || c.is_numeric()))
        .count();
    if punc == 0 {
        return None;
    }
    let nl = s[(space + punc)..]
        .iter()
        .take_while(|&&(_, c)| c == '\r' || c == '\n')
        .count();
    Some(space + punc + nl)
}

/// \s*[\r\n]+
fn newlines(s: &[(usize, char)]) -> Option<usize> {
    let run = s.iter().take_while(|&&(_, c)| c.is_whitespace()).count();
    let last = s[..run]
        .iter()
        .rposition(|&(_, c)| c == '\r' || c == '\n')?;
    Some(last + 1)
}

/// \s+(?!\S)
fn space(s: &[(usize, char)]) -> Option<usize> {
    let run = s.iter().take_while(|&&(_, c)| c.is_whitespace()).count();
    if run == s.len() {
        (run > 0).then_some(run)
    } else {
        (run > 1).then(|| run - 1)
    }
}

/// \s+
fn whitespace(s: &[(usize, char)]) -> Option<usize> {
    let run = s.iter().take_while(|&&(_, c)| c.is_whitespace()).count();
    (run > 0).then_some(run)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokenizer(text: &str) -> Tokenizer {
        Tokenizer::from_json(&json::parse(text).unwrap()).unwrap()
    }

    fn char_indices(s: &str) -> Vec<(usize, char)> {
        s.char_indices().collect()
    }

    #[test]
    fn from_json_places_tokens_by_id() {
        let text = r#"{"model":{"vocab":{"b":1,"a":0}}}"#;
        let tokenizer = tokenizer(text);
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
    fn merge_prefers_earlier_merges() {
        let text = r#"{"model":{"vocab":{"a":0,"b":1,"c":2,"bc":3,"ab":4}}}"#;
        let tokenizer = tokenizer(text);
        assert_eq!(tokenizer.merge("abc"), [0, 3]);
    }

    #[test]
    fn merge_chains_merges() {
        let text = r#"{"model":{"vocab":{"a":0,"b":1,"c":2,"ab":3,"abc":4}}}"#;
        let tokenizer = tokenizer(text);
        assert_eq!(tokenizer.merge("abc"), [4]);
        assert_eq!(tokenizer.merge("cab"), [2, 3]);
    }

    #[test]
    fn merge_spells_bytes_first() {
        let text = r#"{"model":{"vocab":{"Ġ":0,"a":1,"Ġa":2}}}"#;
        let tokenizer = tokenizer(text);
        assert_eq!(tokenizer.merge(" a"), [2]);
    }

    #[test]
    fn decode_reverses_spellings() {
        let text = r#"{"model":{"vocab":{"Ġhello":0,"Ã©":1,"!":2}}}"#;
        let tokenizer = tokenizer(text);
        assert_eq!(tokenizer.decode(&[0, 1, 2]), " helloé!");
    }

    #[test]
    fn decode_replaces_invalid_utf8() {
        let text = r#"{"model":{"vocab":{"Ã":0}}}"#;
        let tokenizer = tokenizer(text);
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

    #[test]
    fn split_matches_example() {
        assert_eq!(
            split("We've 12 cats!"),
            ["We", "'ve", " ", "1", "2", " cats", "!"]
        );
    }

    #[test]
    fn split_keeps_the_last_space_for_the_word() {
        assert_eq!(split("  Hello"), [" ", " Hello"]);
    }

    #[test]
    fn split_glues_newlines_to_punctuation() {
        assert_eq!(split("!\n"), ["!\n"]);
        assert_eq!(split("a\n\nb"), ["a", "\n\n", "b"]);
    }

    #[test]
    fn split_tiles_the_text() {
        for text in ["We've 12 cats!", "  Hello  ", "a\r\n\r\nb", " é’…\t9", ""] {
            assert_eq!(split(text).concat(), text, "{text}");
        }
    }

    #[test]
    fn contraction_takes_the_seven_tails() {
        for tail in ["'s", "'t", "'m", "'d", "'re", "'ve", "'ll"] {
            assert_eq!(contraction(&char_indices(tail)), Some(tail.len()), "{tail}");
        }
        assert_eq!(contraction(&char_indices("'S")), Some(2));
        assert_eq!(contraction(&char_indices("'r")), None);
    }

    #[test]
    fn word_takes_letters_and_their_prefix() {
        assert_eq!(word(&char_indices("hello")), Some(5));
        assert_eq!(word(&char_indices(" hello")), Some(6));
        assert_eq!(word(&char_indices(" Hello")), Some(6));
        assert_eq!(word(&char_indices("42")), None);
    }

    #[test]
    fn number_takes_one_digit() {
        assert_eq!(number(&char_indices("a")), None);
        assert_eq!(number(&char_indices("1")), Some(1));
        assert_eq!(number(&char_indices("12")), Some(1));
    }

    #[test]
    fn punctuation_takes_a_run_and_trailing_newlines() {
        assert_eq!(punctuation(&char_indices("")), None);
        assert_eq!(punctuation(&char_indices("1")), None);
        assert_eq!(punctuation(&char_indices(".")), Some(1));
        assert_eq!(punctuation(&char_indices("..")), Some(2));
        assert_eq!(punctuation(&char_indices(".1")), Some(1));
        assert_eq!(punctuation(&char_indices(".1\n")), Some(1));
        assert_eq!(punctuation(&char_indices("..\n\r.")), Some(4));
        assert_eq!(punctuation(&char_indices(" ..\n\r.")), Some(5));
        assert_eq!(punctuation(&char_indices(" 1..\n.")), None);
        assert_eq!(punctuation(&char_indices("\n")), None);
        assert_eq!(punctuation(&char_indices(" \n.")), None);
    }

    #[test]
    fn newlines_end_at_the_last_newline() {
        assert_eq!(newlines(&char_indices(" ")), None);
        assert_eq!(newlines(&char_indices("1\n")), None);
        assert_eq!(newlines(&char_indices("\n")), Some(1));
        assert_eq!(newlines(&char_indices(" \n")), Some(2));
        assert_eq!(newlines(&char_indices(" \n\r\n")), Some(4));
        assert_eq!(newlines(&char_indices(" \n \n ")), Some(4));
    }

    #[test]
    fn space_gives_the_last_space_back() {
        assert_eq!(space(&char_indices("")), None);
        assert_eq!(space(&char_indices("a")), None);
        assert_eq!(space(&char_indices(" a")), None);
        assert_eq!(space(&char_indices("  a")), Some(1));
        assert_eq!(space(&char_indices("  ")), Some(2));
        assert_eq!(space(&char_indices(" ")), Some(1));
    }

    #[test]
    fn whitespace_consumes_spaces() {
        assert_eq!(whitespace(&char_indices("")), None);
        assert_eq!(whitespace(&char_indices("a ")), None);
        assert_eq!(whitespace(&char_indices(" ")), Some(1));
        assert_eq!(whitespace(&char_indices("  a")), Some(2));
    }
}
