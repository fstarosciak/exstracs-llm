use std::fmt;

#[derive(Debug)]
pub enum Error {
    Io(std::io::Error),
    Csv(csv::Error),
    Json(serde_json::Error),
    Data(String),
    Config(String),
}

pub type Result<T> = std::result::Result<T, Error>;

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(error) => write!(f, "I/O error: {error}"),
            Error::Csv(error) => write!(f, "CSV error: {error}"),
            Error::Json(error) => write!(f, "JSON error: {error}"),
            Error::Data(message) => write!(f, "invalid data: {message}"),
            Error::Config(message) => write!(f, "invalid configuration: {message}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io(error) => Some(error),
            Error::Csv(error) => Some(error),
            Error::Json(error) => Some(error),
            Error::Data(_) | Error::Config(_) => None,
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(error: std::io::Error) -> Self {
        Error::Io(error)
    }
}

impl From<csv::Error> for Error {
    fn from(error: csv::Error) -> Self {
        Error::Csv(error)
    }
}

impl From<serde_json::Error> for Error {
    fn from(error: serde_json::Error) -> Self {
        Error::Json(error)
    }
}
