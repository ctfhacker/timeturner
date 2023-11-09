use clap::Parser;
use timeturner::Debugger;

use std::io::Write;
use std::path::PathBuf;

#[derive(Parser, Debug)]
struct Args {
    /// Text trace to rerun
    filename: PathBuf,
}

fn main() -> Result<(), timeturner::Error> {
    let args = Args::parse();
    let mut dbg = Debugger::from_file(&args.filename)?;
    let mut test = dbg.clone();

    println!(
        "Size: {} MB Entries: {} Bytes/Entry: {}",
        dbg.size() as f64 / 1024 as f64 / 1024 as f64,
        dbg.entries,
        dbg.size() as f64 / dbg.entries as f64
    );

    let offset = 12000;

    // let mut commands = Vec::new();
    dbg.goto_index(10 * offset);

    for next_index in [0, 9, 1, 8, 1, 9] {
        let next_index = next_index * offset;
        println!("GOING {} to {next_index}", dbg.index);

        dbg.goto_index(next_index);

        test.reset();
        test.goto_index(next_index);

        assert!(dbg.index == next_index);
        assert!(dbg.index == test.index);
        test.memory.diff(&dbg.memory);
    }

    let _ = dbg.exec_command("stats");

    Ok(())
}

/// Drop into an interactive terminal for this debugger
pub fn interactive(dbg: &mut Debugger) -> Result<(), ()> {
    loop {
        dbg.context_at(dbg.index);
        print!("(timeturner:{}) ", dbg.index);
        std::io::stdout().flush().unwrap();

        let mut command = String::new();

        std::io::stdin()
            .read_line(&mut command)
            .expect("Failed to get command");

        dbg.exec_command(command.trim())?;
    }
}
