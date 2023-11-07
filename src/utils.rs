//! Utilities for rerun

use crate::{Error, Timer};

/// Parse a hex encoded string and return the u64 representation
pub fn parse_hex_string(input: &str) -> Result<u64, crate::Error> {
    timeloop::scoped_timer!(Timer::ParseHexString);

    u64::from_str_radix(&input.replace("0x", ""), 16).map_err(Error::ParseIntError)
}
