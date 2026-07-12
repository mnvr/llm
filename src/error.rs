use std::error::Error;
use std::fmt;

pub struct AnyError(Box<dyn Error>);

impl fmt::Debug for AnyError {
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

impl<E: Error + 'static> From<E> for AnyError {
    fn from(e: E) -> AnyError {
        AnyError(Box::new(e))
    }
}
