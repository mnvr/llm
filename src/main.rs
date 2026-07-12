use std::error::Error;
use std::fs;

use llm::config::Config;
use llm::json;

fn main() -> Result<(), Box<dyn Error>> {
    if let Err(e) = run() {
        report(e.as_ref());
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let text = fs::read_to_string("models/qwen3-4b-base/config.json")?;
    let config = Config::from_json(&json::parse(&text)?);
    println!("{config:#?}");
    Ok(())
}

fn report(err: &dyn Error) {
    eprintln!("error: {err}");
    let mut source = err.source();
    while let Some(e) = source {
        eprintln!("caused by: {e}");
        source = e.source()
    }
}
