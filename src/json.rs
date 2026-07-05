use std::fmt;

pub enum Json {
    Null,
    Bool(bool),
    Number(f64),
    String(String),
    Array(Vec<Json>),
    Object(Vec<(String, Json)>),
}

impl fmt::Display for Json {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Json::Null => write!(f, "null"),
            Json::Bool(b) => write!(f, "{b}"),
            Json::Number(n) => write!(f, "{n}"),
            Json::String(s) => write_string(f, s),
            Json::Array(items) => {
                write!(f, "[")?;
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        write!(f, ",")?;
                    }
                    write!(f, "{item}")?;
                }
                write!(f, "]")
            }
            Json::Object(members) => {
                write!(f, "{{")?;
                for (i, (key, value)) in members.iter().enumerate() {
                    if i > 0 {
                        write!(f, ",")?;
                    }
                    write_string(f, key)?;
                    write!(f, ":{value}")?;
                }
                write!(f, "}}")
            }
        }
    }
}

fn write_string(f: &mut fmt::Formatter<'_>, s: &str) -> fmt::Result {
    write!(f, "\"")?;
    for c in s.chars() {
        match c {
            '"' => write!(f, "\\\"")?,
            '\\' => write!(f, "\\\\")?,
            '\n' => write!(f, "\\n")?,
            '\r' => write!(f, "\\r")?,
            '\t' => write!(f, "\\t")?,
            c if (c as u32) < 0x20 => write!(f, "\\u{:04x}", c as u32)?,
            c => write!(f, "{c}")?,
        }
    }
    write!(f, "\"")
}

#[derive(Debug)]
pub struct ParseError {
    pub pos: usize,
    pub msg: &'static str,
}

struct Parser<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl Parser<'_> {
    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn err(&self, msg: &'static str) -> ParseError {
        ParseError { pos: self.pos, msg }
    }

    fn skip_ws(&mut self) {
        while let Some(b' ' | b'\t' | b'\n' | b'\r') = self.peek() {
            self.pos += 1;
        }
    }

    fn expect(&mut self, s: &'static str) -> Result<(), ParseError> {
        if self.bytes[self.pos..].starts_with(s.as_bytes()) {
            self.pos += s.len();
            Ok(())
        } else {
            Err(self.err(s))
        }
    }

    fn value(&mut self) -> Result<Json, ParseError> {
        match self.peek() {
            Some(b'n') => {
                self.expect("null")?;
                Ok(Json::Null)
            }
            Some(b't') => {
                self.expect("true")?;
                Ok(Json::Bool(true))
            }
            Some(b'f') => {
                self.expect("false")?;
                Ok(Json::Bool(false))
            }
            _ => Err(self.err("a JSON value")),
        }
    }
}

pub fn parse(s: &str) -> Result<Json, ParseError> {
    let mut p = Parser {
        bytes: s.as_bytes(),
        pos: 0,
    };
    p.skip_ws();
    let value = p.value()?;
    p.skip_ws();
    if p.pos < p.bytes.len() {
        return Err(p.err("end of input"));
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_prints_compact_json() {
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
        assert_eq!(
            format!("{doc}"),
            r#"{"model_type":"qwen3","hidden_size":2560,"tie_word_embeddings":true,"rope_scaling":null,"sizes":[0.6,1.7,4]}"#
        );
    }

    #[test]
    fn parse_accepts_literals() {
        assert_eq!(parse("null").unwrap().to_string(), "null");
        assert_eq!(parse("  true ").unwrap().to_string(), "true");
        assert_eq!(parse("false").unwrap().to_string(), "false");
    }

    #[test]
    fn parse_rejects_malformed_input() {
        assert!(parse("nul").is_err());
        assert!(parse("").is_err());
        assert!(parse("true false").is_err());
    }

    #[test]
    fn parse_reports_error_position_and_msg() {
        let e = parse("truefalse");
        match e {
            Err(ParseError { pos, msg }) => {
                assert_eq!(pos, 4);
                assert_eq!(msg, "end of input")
            }
            _ => assert!(false),
        }
    }
}
