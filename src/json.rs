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
            Some(b'"') => self.string().map(Json::String),
            Some(b'[') => self.array(),
            Some(b'{') => self.object(),
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
            return Err(ParseError {
                pos: start,
                msg: "a representable number",
            });
        }
        Ok(Json::Number(n))
    }

    fn string(&mut self) -> Result<String, ParseError> {
        self.expect("\"")?;
        let mut out = String::new();
        let mut run = self.pos;
        loop {
            let Some(b) = self.peek() else {
                return Err(self.err("a closing quote"));
            };
            match b {
                b'"' | b'\\' => {
                    out.push_str(str::from_utf8(&self.bytes[run..self.pos]).unwrap());
                    self.pos += 1;
                    if b == b'"' {
                        return Ok(out);
                    }
                    out.push(self.escape()?);
                    run = self.pos;
                }
                0x00..=0x1f => return Err(self.err("an escaped control character")),
                _ => self.pos += 1,
            }
        }
    }

    fn escape(&mut self) -> Result<char, ParseError> {
        let c = match self.peek() {
            Some(b'"') => '"',
            Some(b'\\') => '\\',
            Some(b'/') => '/',
            Some(b'b') => '\u{8}',
            Some(b'f') => '\u{c}',
            Some(b'n') => '\n',
            Some(b'r') => '\r',
            Some(b't') => '\t',
            Some(b'u') => {
                self.pos += 1;
                return self.unicode_escape();
            }
            _ => return Err(self.err("an escape character")),
        };
        self.pos += 1;
        Ok(c)
    }

    fn unicode_escape(&mut self) -> Result<char, ParseError> {
        let start = self.pos;
        let hi = self.hex4()?;
        let code = if (0xD800..=0xDBFF).contains(&hi) {
            self.expect("\\u")?;
            let lo = self.hex4()?;
            if !(0xDC00..=0xDFFF).contains(&lo) {
                return Err(ParseError {
                    pos: start,
                    msg: "a low surrogate",
                });
            }
            0x10000 + ((hi - 0xD800) << 10) + (lo - 0xDC00)
        } else {
            hi
        };
        char::from_u32(code).ok_or(ParseError {
            pos: start,
            msg: "a Unicode scalar value",
        })
    }

    fn hex4(&mut self) -> Result<u32, ParseError> {
        let mut v = 0;
        for _ in 0..4 {
            let d = match self.peek() {
                Some(b @ b'0'..=b'9') => u32::from(b - b'0'),
                Some(b @ b'a'..=b'f') => u32::from(b - b'a' + 10),
                Some(b @ b'A'..=b'F') => u32::from(b - b'A' + 10),
                _ => return Err(self.err("a hex digit")),
            };
            self.pos += 1;
            v = v << 4 | d;
        }
        Ok(v)
    }

    fn array(&mut self) -> Result<Json, ParseError> {
        self.pos += 1;
        let mut items = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b']') {
            self.pos += 1;
            return Ok(Json::Array(items));
        }
        loop {
            self.skip_ws();
            items.push(self.value()?);
            self.skip_ws();
            match self.peek() {
                Some(b',') => self.pos += 1,
                Some(b']') => {
                    self.pos += 1;
                    return Ok(Json::Array(items));
                }
                _ => return Err(self.err("',' or ']'")),
            }
        }
    }

    fn object(&mut self) -> Result<Json, ParseError> {
        self.pos += 1;
        let mut members = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b'}') {
            self.pos += 1;
            return Ok(Json::Object(members));
        }
        loop {
            self.skip_ws();
            let key = self.string()?;
            self.skip_ws();
            self.expect(":")?;
            self.skip_ws();
            let value = self.value()?;
            members.push((key, value));
            self.skip_ws();
            match self.peek() {
                Some(b',') => self.pos += 1,
                Some(b'}') => {
                    self.pos += 1;
                    return Ok(Json::Object(members));
                }
                _ => return Err(self.err("',' or '}'")),
            }
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

impl Json {
    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Object(members) => members.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Json::Number(n) => Some(*n),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Json::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Json::Bool(b) => Some(*b),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[Json]> {
        match self {
            Json::Array(items) => Some(items),
            _ => None,
        }
    }

    pub fn as_object(&self) -> Option<&[(String, Json)]> {
        match self {
            Json::Object(members) => Some(members),
            _ => None,
        }
    }
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
        let Err(e) = parse("truefalse") else {
            panic!("expected error")
        };
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

    #[test]
    fn parse_accepts_strings() {
        for (input, expected) in [
            (r#""hello""#, "hello"),
            (r#""""#, ""),
            (r#""a\"b\\c\/d""#, "a\"b\\c/d"),
            (r#""tab\there""#, "tab\there"),
            (r#""\u0041""#, "A"),
            (r#""caf\u00e9""#, "café"),
            (r#""\ud83d\ude00""#, "😀"),
            (r#""直接""#, "直接"),
        ] {
            let Json::String(s) = parse(input).unwrap() else {
                panic!("{input}")
            };
            assert_eq!(s, expected);
        }
    }

    #[test]
    fn parse_rejects_malformed_strings() {
        for s in [
            r#"""#,
            r#""\x""#,
            r#""\u12""#,
            r#""\ud800""#,
            r#""\ud8000\u0041""#,
            "\"\n\"",
        ] {
            assert!(parse(s).is_err(), "{s}");
        }
    }

    #[test]
    fn parse_accepts_containers() {
        for (input, expected) in [
            ("[]", "[]"),
            ("{}", "{}"),
            ("[ 1 , true , \"x\" ]", r#"[1,true,"x"]"#),
            (
                r#"{ "a" : [1, {"b": null}] , "c" : -2.5e3 }"#,
                r#"{"a":[1,{"b":null}],"c":-2500}"#,
            ),
        ] {
            assert_eq!(parse(input).unwrap().to_string(), expected, "{input}");
        }
    }

    #[test]
    fn parse_rejects_malformed_containers() {
        for s in [
            "[",
            "]",
            "[1,]",
            "[1 2]",
            "[[]",
            "{",
            r#"{"a"}"#,
            r#"{"a":}"#,
            r#"{"a":1,}"#,
            r#"{a:1}"#,
        ] {
            assert!(parse(s).is_err(), "{s}");
        }
    }
}
