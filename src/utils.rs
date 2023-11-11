//! Utilities for rerun

use crate::Error;

/// Parse a hex encoded string and return the u64 representation
pub fn parse_hex_string(input: &str) -> Result<u64, crate::Error> {
    timeloop::scoped_timer!(crate::Timer::ParseHexString);

    u64::from_str_radix(&input.replace("0x", "").replace("\n", ""), 16)
        .map_err(Error::ParseIntError)
}
