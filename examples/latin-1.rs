fn main() {
    for b in 0..=u8::MAX {
        let c = char::from(b);
        let c = if c.is_control() { '○' } else { c };
        print!("{b:>3} {c}  ");
        if b % 10 == 9 {
            println!();
        }
    }
    println!();
}
