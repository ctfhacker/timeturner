use clap::Parser;
use rand::RngCore;
use timeturner::Debugger;

use std::path::PathBuf;

#[derive(Parser, Debug)]
struct Args {
    /// Text trace to rerun
    filename: PathBuf,
}

fn main() -> Result<(), timeturner::Error> {
    env_logger::init();
    let args = Args::parse();

    let start = std::time::Instant::now();
    let mut dbg = Debugger::from_file(&args.filename)?;
    println!("Reading file took: {:?}", start.elapsed());

    /*
    for _ in 0..1 {
        single_forward_step_test(&mut dbg);
        single_backward_step_test(&mut dbg);
        goto_random_index_test(&mut dbg);
    }

    dbg.print_stats();
    */

    for i in [0, 1, 2, 3, 2, 1, 0] {
        dbg.goto_index(i);
        dbg.hexdump(0xcff70, 0x20);
        dbg.print_context();
    }

    Ok(())
}

/// Drop into an interactive terminal for this debugger
pub fn single_forward_step_test(dbg: &mut Debugger) {
    const ITERS: usize = 10000;
    let mut single_step = std::time::Duration::from_secs(0);
    let mut worst_single_step = std::time::Duration::from_secs(0);
    let mut best_single_step = std::time::Duration::from_secs(9999);

    for _ in 0..ITERS {
        let start = std::time::Instant::now();

        dbg.step_forward();

        let elapsed = start.elapsed();
        if elapsed > worst_single_step {
            worst_single_step = elapsed;
        }
        if elapsed < best_single_step {
            best_single_step = elapsed;
        }

        single_step += elapsed;
    }

    println!("--- Single Step Forward ---");
    println!("Best:  {:.4?}", best_single_step);
    println!("Avg:   {:.4?}", single_step / ITERS.try_into().unwrap());
    println!("Worst: {:.4?}", worst_single_step);
}

pub fn single_backward_step_test(dbg: &mut Debugger) {
    const ITERS: usize = 10000;
    let mut avg = std::time::Duration::from_secs(0);
    let mut worst = std::time::Duration::from_secs(0);
    let mut best = std::time::Duration::from_secs(9999);

    dbg.goto_index(dbg.entries / 2);

    for _ in 0..ITERS {
        let start = std::time::Instant::now();

        dbg.step_backward();

        let elapsed = start.elapsed();
        if elapsed > worst {
            worst = elapsed;
        }
        if elapsed < best {
            best = elapsed;
        }

        avg += elapsed;
    }

    println!("--- Single Step Backward ---");
    println!("Best:  {:.4?}", best);
    println!("Avg:   {:.4?}", avg / ITERS.try_into().unwrap());
    println!("Worst: {:.4?}", worst);
}

pub fn goto_random_index_test(dbg: &mut Debugger) {
    const ITERS: usize = 25;
    let mut worst_move = (0, 0);
    let mut best_move = (0, 0);

    let mut avg = std::time::Duration::from_secs(0);
    let mut worst = std::time::Duration::from_secs(0);
    let mut best = std::time::Duration::from_secs(9999);
    let mut rng = rand::thread_rng();

    dbg.goto_index(dbg.entries / 2);

    for _ in 0..ITERS {
        let start = std::time::Instant::now();
        let starting_index = dbg.index;

        let next_index = rng.next_u32() % dbg.entries;
        dbg.goto_index(next_index);

        let elapsed = start.elapsed();
        if elapsed > worst {
            worst = elapsed;
            worst_move = (starting_index, next_index);
        }
        if elapsed < best {
            best = elapsed;
            best_move = (starting_index, next_index);
        }

        avg += elapsed;
    }

    println!("--- Random index ---");
    println!("Best:  {:.4?} {best_move:?}", best);
    println!("Avg:   {:.4?}", avg / ITERS.try_into().unwrap());
    println!("Worst: {:.4?} {worst_move:?}", worst);
}
