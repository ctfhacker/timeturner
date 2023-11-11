//! Test to confirm saving and restoring debugger state from disk

use std::path::Path;
use timeturner::Debugger;

#[test]
fn main() -> Result<(), timeturner::Error> {
    let Ok(dir) = std::env::var("CARGO_MANIFEST_DIR") else {
        panic!("Failed to get CARGO_MANIFEST_DIR");
    };

    // Get the test log
    let filename = Path::new(&dir).join("tests").join("trace.0.log.100k");

    // Get the debugger that will be used for testing
    let dbg = Debugger::from_file(&filename)?;

    let save_file = Path::new("/tmp/.testfile.gz");
    dbg.save(save_file);

    let dbg2 = Debugger::restore(save_file);
    std::fs::remove_file(&save_file).unwrap();

    assert!(dbg == dbg2.unwrap_or_else(|e| panic!("Failed to read from disk: {e:?}")));

    Ok(())
}
