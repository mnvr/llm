pub fn byte_char(b: u8) -> char {
    let c = match b {
        0x21..=0x7e | 0xa1..=0xac | 0xae..=0xff => u32::from(b),
        0x00..=0x20 => 0x100 + u32::from(b),
        0x7f..=0xa0 => 0x121 + u32::from(b - 0x7f),
        0xad => 0x143,
    };
    char::from_u32(c).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

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
    fn byte_char_is_injective() {
        let spellings: HashSet<char> = (0..=u8::MAX).map(byte_char).collect();
        assert_eq!(spellings.len(), 256);
    }
}
