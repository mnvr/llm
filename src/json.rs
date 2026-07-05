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
            Some(b'-' | b'0'..=b'9') => self.number(),
            _ => Err(self.err("a JSON value")),
        }
    }

    fn digits(&mut self) -> Result<(), ParseError> {
        let start = self.pos;
        while self.peek().is_some_and(|b| b.is_ascii_digit()) {
            self.pos += 1;
        }
        if self.pos == start {
            Err(self.err("a digit"))
        } else {
            Ok(())
        }
    }

    fn number(&mut self) -> Result<Json, ParseError> {
        let start = self.pos;
        if self.peek() == Some(b'-') {
            self.pos += 1;
        }
        if self.peek() == Some(b'0') {
            self.pos += 1;
        } else {
            self.digits()?;
        }
        if self.peek() == Some(b'.') {
            self.pos += 1;
            self.digits()?;
        }
        if let Some(b'e' | b'E') = self.peek() {
            self.pos += 1;
            if let Some(b'+' | b'-') = self.peek() {
                self.pos += 1;
            }
            self.digits()?;
        }
        let s = str::from_utf8(&self.bytes[start..self.pos]).unwrap();
        let n: f64 = s.parse().unwrap();
        if !n.is_finite() {
            return Err(ParseError { pos: start, msg: "a representable number" });
        }
        Ok(Json::Number(n))
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
        let Err(e) = parse("truefalse") else { panic!("expected error") };
        assert_eq!(e.pos, 4);
        assert_eq!(e.msg, "end of input")
    }

    #[test]
    fn parse_accepts_numbers() {
        for (input, expected) in [
            ("0", "0"),
            ("-0", "-0"),
            ("2560", "2560"),
            ("0.6", "0.6"),
            ("1e-06", "0.000001"),
            ("5000000.0", "5000000"),
            ("-12.5e3", "-12500"),
        ] {
            assert_eq!(parse(input).unwrap().to_string(), expected);
        }
    }

    #[test]
    fn parse_rejects_malformed_numbers() {
        for s in ["01", ".5", "-", "1.", "1e", "+1", "1e+", "0x10", "1e309"] {
            assert!(parse(s).is_err(), "{s}");
        }
    }
}
