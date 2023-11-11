//! Test to ensure that going between instruction traces is the same
//! execution state as if the debugger was executed from the beginning.

use timeturner::{memory::Memory, Debugger};

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

const ITERS: &[usize] = &[0, 9, 0, 5, 9, 4, 1, 3, 5, 7, 9, 4, 0, 9, 0];

/// Generate the snapshot locations based on the size of the test trace
fn get_test_indexes(entries: usize) -> [usize; ITERS.len()] {
    // Calculate the largest power of ten for the number of entries
    let mut multiplier = 1_usize;
    for i in 1..32 {
        let result = 10_usize.pow(i);
        if entries <= result {
            multiplier = result / 10;
            break;
        }
    }

    // Create the actual snapshot indexes based on the new multiplier
    let mut result = [0; ITERS.len()];
    let mut index = 0;
    loop {
        if index >= ITERS.len() {
            break;
        }

        result[index] = ITERS[index] * multiplier;
        index += 1;
    }

    result
}

/// Get memory snapshots for the indexes in the ITERS.
/// These will be compared during the test to confirm the memory
/// is reverted properly.
fn get_snapshots(file: &Path) -> BTreeMap<usize, Memory> {
    let mut dbg = Debugger::from_file(file).unwrap();

    // Get all of the unique indexes in order
    let locations = get_test_indexes(dbg.entries)
        .iter()
        .copied()
        .collect::<BTreeSet<usize>>();
    let mut locations = locations.iter().collect::<Vec<_>>();
    locations.sort();

    // Go to each index in order and take a copy of the current memory state
    let mut result = BTreeMap::new();
    for index in locations {
        // Go forward to the next index
        dbg.goto_index(*index);

        // Clone the current memory state to check in the test
        result.insert(*index, dbg.memory.clone());
    }

    result
}

#[test]
fn main() -> Result<(), timeturner::Error> {
    let Ok(dir) = std::env::var("CARGO_MANIFEST_DIR") else {
        panic!("Failed to get CARGO_MANIFEST_DIR");
    };

    // Get the test log
    let filename = Path::new(&dir).join("tests").join("trace.0.log.100k");

    // Get the memory snapshots we are testing against
    let test_snapshots = get_snapshots(&filename);

    // Get the debugger that will be used for testing
    let mut dbg = Debugger::from_file(&filename)?;

    // Go to each entry in the test and check that the memory matches
    for next_index in get_test_indexes(dbg.entries) {
        dbg.goto_index(next_index);
        test_snapshots[&next_index].diff(&dbg.memory);
    }

    Ok(())
}
