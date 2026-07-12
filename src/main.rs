use llm::config::Config;
use llm::error::AnyError;

fn main() -> Result<(), AnyError> {
    let config = Config::load("models/qwen3-4b-base/config.json")?;
    println!("{config:#?}");
    Ok(())
}
