use llm::config::Config;
use llm::error::Report;

fn main() -> Result<(), Report> {
    let config = Config::load("models/qwen3-4b-base/config.json")?;
    println!("{config:#?}");
    Ok(())
}
