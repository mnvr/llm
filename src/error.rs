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
