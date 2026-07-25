use std::error::Error;
use std::fmt;

pub struct Report(Box<dyn Error>);

impl fmt::Debug for Report {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)?;
        let mut source = self.0.source();
        while let Some(e) = source {
            write!(f, "\ncaused by: {e}")?;
            source = e.source();
        }
        Ok(())
    }
}

impl<E: Error + 'static> From<E> for Report {
    fn from(e: E) -> Report {
        Report(Box::new(e))
    }
}

#[derive(Debug)]
pub struct LoadError {
    path: String,
    source: Box<dyn Error>,
}

impl LoadError {
    pub fn new(path: &str, source: Box<dyn Error>) -> LoadError {
        LoadError {
            path: path.to_string(),
            source,
        }
    }
}

impl fmt::Display for LoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "could not load {}", self.path)
    }
}

impl Error for LoadError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        Some(self.source.as_ref())
    }
}
