use clap::Parser;
use timeturner::{Debugger, InstrIndex, Register};

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
    ParseError(String),
    Io(std::io::Error),
    Timeturner(timeturner::Error),
    MissingCommandArgument,
}

#[derive(Debug, Copy, Clone)]
pub enum Breakpoint {
    Write(u64),
    Read(u64),
    Exec(u64),
}

#[derive(Debug)]
pub enum Execution {
    NewContext,
    NoNewContext,
    Quit,
}

fn main() -> Result<(), Error> {
    env_logger::init();
    let args = CliArgs::parse();

    let start = std::time::Instant::now();
    let mut dbg = Debugger::from_file(&args.filename).map_err(Error::Timeturner)?;
    println!("Reading file took: {:?}", start.elapsed());

    let mut command = String::new();
    let mut last_command = String::from("n");
    let mut breakpoints = Vec::new();

    dbg.print_context();
    print!("(tt:{}) ", dbg.index);
    std::io::stdout().flush().unwrap();

    loop {
        command.clear();

        std::io::stdin()
            .read_line(&mut command)
            .map_err(Error::Io)?;

        let start = std::time::Instant::now();

        // If there was just a new line, repeat the last command
        let result = if command == "\n" {
            match exec_command(&mut dbg, &last_command, &mut breakpoints) {
                Ok(result) => result,
                Err(e) => {
                    println!("ERROR: {e:?}");
                    Execution::NoNewContext
                }
            }
        } else {
            match exec_command(&mut dbg, &command, &mut breakpoints) {
                Ok(result) => {
                    last_command = command.clone();
                    result
                }
                Err(e) => {
                    println!("ERROR: {e:?}");
                    Execution::NoNewContext
                }
            }
        };

        println!("Command took: {:?}", start.elapsed());

        match result {
            Execution::Quit => break,
            Execution::NoNewContext => {}
            Execution::NewContext => {
                dbg.print_context();
            }
        }

        print!("(tt:{}) ", dbg.index);
        std::io::stdout().flush().unwrap();
    }

    Ok(())
}

/// Parse a number. If prefixed with '0x', parse as hex, otherwise as decimal
pub fn parse_number(num: &str) -> Result<u64, Error> {
    // Parse the number if given one
    if let Some(val) = num.strip_prefix("0x") {
        u64::from_str_radix(val, 16).map_err(|_| Error::ParseError(num.to_string()))
    } else {
        num.parse::<u64>()
            .map_err(|_| Error::ParseError(num.to_string()))
    }
}

/// Parse a command argument from either a register or number
pub fn parse_argument(arg: Option<&str>, dbg: &Debugger) -> Result<u64, Error> {
    let Some(arg) = arg else {
        return Err(Error::MissingCommandArgument);
    };

    if let Some(reg) = Register::from_str(arg) {
        Ok(dbg.get_register(reg).0)
    } else {
        parse_number(arg)
    }
}

pub fn exec_command(
    dbg: &mut Debugger,
    command: &str,
    breakpoints: &mut Vec<Breakpoint>,
) -> Result<Execution, Error> {
    let mut args = command.trim().split(' ');
    let command = args.next().unwrap();

    let execution = match command {
        "g" => {
            let end_index = match args.next() {
                Some("end") | None => {
                    // 'g' by itself just runs to the end
                    dbg.entries - 1
                }
                Some(num) => {
                    // Parse the index go to
                    parse_number(num)? as InstrIndex
                }
            };

            dbg.goto_index(end_index);
            Execution::NewContext
        }
        "g-" => {
            let mut index = 0;

            dbg.goto_index(index);
            Execution::NewContext
        }
        "memreads" => {
            let addr = parse_argument(args.next(), dbg)?;
            let reads = dbg.get_memory_reads(addr);
            println!("MEM READS {addr:#x} | {reads:?}");

            Execution::NoNewContext
        }
        /*
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
        "bl" => {
            if breakpoints.is_empty() {
                println!("No breakpoints set to display.")
            } else {
                for bp in breakpoints {
                    println!("{bp:x?}");
                }
            }

            Execution::NoNewContext
        }
        /*
        "ba" => {
            let Some(arg) = args.next() else {
                println!("ERROR: {command} <ADDRESS|REGISTER>");
                return Ok(Execution::Usage);
            };

            Some(num) => arg {
                let addr = parse_number(num) as u64;

                // Add a write breakpoint if there are any writes for this address
                if dbg.memory_writes.contains_key(&addr) {
                    breakpoints.push(Breakpoint::Write(addr));
                } else {
                    println!("No memory writes available for {addr:#x}");
                }

                // Add a read breakpoint if there are any reads for this address
                if dbg.memory_reads.contains_key(&addr) {
                    breakpoints.push(Breakpoint::Read(addr));
                } else {
                    println!("No memory writes available for {addr:#x}");
                }
            }
            _ => {
                println!("ERROR: ba <ADDRESS>");
                return Ok(Execution::Usage);
            }
        },
        "br" => {
            let Some(arg) = args.next() else {
                println!("ERROR: {command} <ADDRESS|REGISTER>");
                return Ok(Execution::Usage);
            };

            if let Some(addr) = parse_argument(arg) {
                // Add a read breakpoint if there are any reads for this address
                if dbg.memory_reads.contains_key(&addr) {
                    breakpoints.push(Breakpoint::Read(addr));
                } else {
                    println!("No memory reads available for {addr:#x}");
                }
            } else {
                println!("ERROR: {command} <ADDRESS|REGISTER>");
                return Ok(Execution::Usage);
            }

        }
        "bp" => {
            let Some(arg) = args.next() else {
                println!("ERROR: {command} <ADDRESS|REGISTER>");
                return Ok(Execution::Usage);
            };

            if let Some(num) = arg1 {
                let addr = parse_number(num) as u64;

                // Add a write breakpoint if there are any writes for this address
                if dbg.rips.contains_key(&addr) {
                    breakpoints.push(Breakpoint::Exec(addr));
                } else {
                    println!("No memory execs available for {addr:#x}");
                }
            } else {
                println!("ERROR: {command} <ADDRESS|REGISTER>");
                return Ok(Execution::Usage);
            }
        }
        */
        "bw" => {
            let addr = match parse_argument(args.next(), dbg) {
                Ok(addr) => addr,
                Err(e) => {
                    println!("ERROR: {e:?}");
                    return Ok(Execution::NoNewContext);
                }
            };

            // Add a write breakpoint if there are any writes for this address
            if dbg.memory_writes.contains_key(&addr) {
                breakpoints.push(Breakpoint::Write(addr));
            } else {
                println!("No memory writes available for {addr:#x}");
            }

            // No need to print the context again
            Execution::NoNewContext
        }
        "next" | "n" => {
            let iters = parse_argument(args.next(), dbg).unwrap_or(1) as InstrIndex;
            let new_index = (dbg.index + iters).min(dbg.entries - 1);
            dbg.goto_index(new_index);
            Execution::NewContext
        }
        "sb" | "s-" => {
            let iters = parse_argument(args.next(), dbg).unwrap_or(1) as InstrIndex;
            let new_index = dbg.index.saturating_sub(iters);
            dbg.goto_index(new_index);
            Execution::NewContext
        }
        "stats" => {
            dbg.print_stats();
            Execution::NoNewContext
        }
        "q" => Execution::Quit,
        x => {
            println!("Unknown command: {x}");
            Execution::NoNewContext
        }
    };

    Ok(execution)
}
