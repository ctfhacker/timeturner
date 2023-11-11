use clap::Parser;
use timeturner::{Debugger, InstrIndex};

use std::io::Write;
use std::path::PathBuf;

#[derive(Parser, Debug)]
struct CliArgs {
    /// Text trace to rerun
    filename: PathBuf,
}

#[derive(Parser, Debug)]
struct CmdArgs {
    /// Go to an index
    #[arg(short, long)]
    go: InstrIndex,
}

#[derive(Debug)]
pub enum Error {
    Io(std::io::Error),
    Timeturner(timeturner::Error),
    MissingCommandArgument,
}

fn main() -> Result<(), Error> {
    env_logger::init();
    let args = CliArgs::parse();

    let start = std::time::Instant::now();
    let mut dbg = Debugger::from_file(&args.filename).map_err(Error::Timeturner)?;
    println!("Reading file took: {:?}", start.elapsed());

    let mut command = String::new();
    let mut last_command = String::from("n");

    loop {
        command.clear();
        dbg.print_context();
        dbg.hexdump(0xcff70, 0x40);
        print!("(tt:{}) ", dbg.index);
        std::io::stdout().flush().unwrap();

        std::io::stdin()
            .read_line(&mut command)
            .map_err(Error::Io)?;

        // If there was just a new line, repeat the last command
        if command == "\n" {
            exec_command(&mut dbg, &last_command)?;
            continue;
        }

        if exec_command(&mut dbg, &command)? {
            break;
        }
        last_command = command.clone();
    }

    Ok(())
}

/// Parse a number. If prefixed with '0x', parse as hex, otherwise as decimal
pub fn parse_number(num: &str) -> InstrIndex {
    // Parse the number if given one
    if let Some(val) = num.strip_prefix("0x") {
        InstrIndex::from_str_radix(val, 16).unwrap()
    } else {
        num.parse::<InstrIndex>().unwrap()
    }
}

pub fn exec_command(dbg: &mut Debugger, command: &str) -> Result<bool, Error> {
    let mut args = command.trim().split(' ');
    let command = args.next().unwrap();
    let arg1 = args.next();

    match command {
        "g" => {
            let index = match arg1.ok_or(Error::MissingCommandArgument)? {
                "end" => dbg.entries - 1,
                num => parse_number(num),
            };

            log::info!("GOTO {index}");
            dbg.goto_index(index);
        }
        /*
        "memreads" => {
            let arg1 = arg1.ok_or(Error::MissingCommandArgument)?;
            let arg1 = arg1.replace("0x", "");
            let address = u64::from_str_radix(&arg1, 16).unwrap();

            let reads = dbg.get_address_reads(address);
            println!("MEM READS {address:#x} | {reads:?}");
        }
        "memwrites" => {
            let arg1 = arg1.ok_or(Error::MissingCommandArgument)?;
            let arg1 = arg1.replace("0x", "");
            let address = u64::from_str_radix(&arg1, 16).unwrap();

            let writes = dbg.get_address_writes(address);
            println!("MEM WRITES {address:#x} | {writes:?}");
        }
        "hexdump" => {
            let arg1 = arg1.ok_or(Error::MissingCommandArgument)?;
            let arg1 = arg1.replace("0x", "");
            let address = u64::from_str_radix(&arg1, 16).unwrap();

            let arg2 = arg2.ok_or(Error::MissingCommandArgument)?;
            let arg2 = arg2.replace("0x", "");
            let n = usize::from_str_radix(&arg2, 16).unwrap();

            dbg.hexdump(address, n);
        }
        */
        "next" | "n" => {
            let iters = match arg1 {
                Some(num) => parse_number(num),
                _ => {
                    // Default to only stepping once
                    1
                }
            };

            for _ in 0..iters {
                dbg.step_forward();
            }
        }
        "sb" => {
            let iters = match arg1 {
                Some(num) => parse_number(num),
                _ => {
                    // Default to only stepping once
                    1
                }
            };

            for _ in 0..iters {
                dbg.step_backward();
            }
        }
        "stats" => dbg.print_stats(),
        "q" => return Ok(true),
        x => println!("Unknown command: {x}"),
    }

    // log::info!("Command took {:?}", start.elapsed());

    Ok(false)
}
