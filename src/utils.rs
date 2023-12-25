//! Utilities for rerun

use crate::Error;

/// Parse a hex encoded string and return the u64 representation
pub fn parse_hex_string(input: &str) -> Result<u64, crate::Error> {
    // Remove the 0x from the beginning of the hex string
    u64::from_str_radix(&input[2..], 16).map_err(|_| Error::ParseIntError(input.to_string()))
}
