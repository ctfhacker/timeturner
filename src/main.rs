use clap::Parser;
use timeturner::Debugger;

use std::path::PathBuf;

pub type Result<T> = std::result::Result<T, timeturner::Error>;

#[derive(Parser, Debug)]
struct Args {
    /// Text trace to rerun
    filename: PathBuf,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let mut dbg = Debugger::from_file(&args.filename)?;

    println!(
        "Size: {} MB Entries: {} Bytes/Entry: {}",
        dbg.size() as f64 / 1024 as f64 / 1024 as f64,
        dbg.entries,
        dbg.size() as f64 / dbg.entries as f64
    );

    let mut commands = Vec::new();

    for _ in 0..10 {
        commands.push("n");
    }

    for command in commands {
        // dbg.print_context();
        let _ = dbg.exec_command(&command);
    }

    for _ in 0..10 {
        dbg.exec_command("sb");
        dbg.print_context();
    }

    dbg.exec_command("stats");

    Ok(())
}
