#![feature(thread_id_value)]

mod colors;
use colors::Colorized;

pub mod memory;
mod utils;

use gzp::{deflate::Gzip, ZBuilder};

use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufRead, BufReader, Read, Write};

use memory::Memory;
use utils::parse_hex_string;

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

/// The maximum number of register entries allowed per trace entry line
/// before heap allocating storage for the register diff
const REGISTER_ENTRIES: usize = 10;

type InstrIndex = u32;

timeloop::impl_enum!(
    #[derive(Debug, Copy, Clone, Eq, PartialEq)]
    pub enum Timer {
        Remaining,
        ParseInput,
        AddDiffs,
        AddMemory,
        SetRegister,
        ParseTrace,
        ReadFile,
        ParseRegister,
        ParseMemory,
        ParseHexString,
        StripPrefix,
        TraceAddDiff,
        TraceAddRegisterDiff,
        TraceAddMemoryDiff,
        ReadInput,
        UpdateMemory,
        UpdateMemoryForward,
        UpdateMemoryBackward,
        UpdateMemoryBackwardCollapseWrites,
        ContextAt,
        PrintContext,
        Hexdump,
        StepForward,
        StepBackward,
        Reset,
        GotoIndex,
        Size,
        ExecCommand,
        GetAddressReads,
        GetAddressWrites,
        GetAddressAccesses,
        TakeMemorySnapshots,
        SaveDebuggerToDisk,
        RestoreDebuggerFromDisk,

        Memory_AllocatePage,
        Memory_SetMemory,
        Memory_GetPageIndex,
        Memory_IsStraddlingPage,
        Memory_Hexdump,
        Memory_Read,
        Memory_SetBytes,
        Memory_SetBytesRead,
        Memory_SetBytesWrite,
        Memory_SetBytesWrite1,
        Memory_SetBytesWrite2,
        Memory_SetBytesCheckByteHistory,
        Memory_SoftReset,
        Memory_SetByte,
        Memory_CheckNewInstrIndex,

        Memory_SetBytes1,
        Memory_SetBytes2,
        Memory_SetBytes3,
        Memory_SetBytes4,
        Memory_SetBytes5,
        Memory_SetBytes6,
        Memory_SetBytes7,
        Memory_SetBytes8,
        Memory_SetBytes9,
    }
);

timeloop::create_profiler!(Timer);

#[derive(Debug)]
pub enum Error {
    Io(std::io::Error),
    ParseIntError(std::num::ParseIntError),
    Bincode(Box<bincode::ErrorKind>),
    InvalidMemoryValue(String),
    AddMemory,
    Quit,
    InvalidCommandFormat,
    InvalidCommandArgument,
    MissingCommandArgument,
}

#[repr(u8)]
#[derive(Debug, Default, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Register {
    #[default]
    Rax,
    Rbx,
    Rcx,
    Rdx,
    Rsi,
    Rdi,
    Rip,
    Rsp,
    Rbp,
    R8,
    R9,
    R10,
    R11,
    R12,
    R13,
    R14,
    R15,

    Eax,
    Ebx,
    Ecx,
    Edx,
    Esi,
    Edi,
    Dx,
    Dl,
    Eip,
    Esp,
    Ebp,
    None,
}

impl Register {
    const fn count() -> usize {
        17
    }

    const fn expanded(&self) -> Register {
        use Register::*;

        match self {
            Rax | Eax => Register::Rax,
            Rbx | Ebx => Register::Rbx,
            Rcx | Ecx => Register::Rcx,
            Rdx | Edx => Register::Rdx,
            Rsi | Esi => Register::Rsi,
            Rdi | Edi | Dx | Dl => Register::Rdi,
            Rip | Eip => Register::Rip,
            Rsp | Esp => Register::Rsp,
            Rbp | Ebp => Register::Rbp,
            R8 => Register::R8,
            R9 => Register::R9,
            R10 => Register::R10,
            R11 => Register::R11,
            R12 => Register::R12,
            R13 => Register::R13,
            R14 => Register::R14,
            R15 => Register::R15,
            _ => unreachable!(),
        }
    }

    const fn bit_mask(&self) -> u64 {
        use Register::*;

        match self {
            Rax | Rbx | Rcx | Rdx | Rsi | Rdi | Rip | Rsp | Rbp | R8 | R9 | R10 | R11 | R12
            | R13 | R14 | R15 => 0x0000_0000_0000_0000,
            Eax | Ebx | Ecx | Edx | Esi | Edi | Eip | Esp | Ebp => 0xffff_ffff_0000_0000,
            _ => unreachable!(),
        }
    }
}

#[repr(u8)]
#[derive(
    Serialize, Deserialize, Debug, Default, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash,
)]
pub enum MemoryAccess {
    #[default]
    None,
    Read,
    Write,
}

/// The set of memory data diffs per line in the trace
#[derive(Serialize, Deserialize, Clone, Default, PartialEq, Eq)]
pub struct MemoryData {
    addresses: Vec<u64>,
    bytes: Vec<Vec<u8>>,
    accesses: Vec<MemoryAccess>,
}

impl std::fmt::Debug for MemoryData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::result::Result<(), std::fmt::Error> {
        let MemoryData {
            addresses,
            bytes,
            accesses,
        } = self;
        for (index, access) in accesses.iter().enumerate() {
            let address = addresses[index];
            let curr_bytes = &bytes[index];

            let operation = match access {
                MemoryAccess::Write => "<-",
                MemoryAccess::Read => "->",
                _ => unreachable!(),
            };

            write!(f, "{access:?} {address:#x} {operation} ")?;

            for byte in curr_bytes {
                write!(f, "{byte:02x}")?;
            }

            write!(f, " ")?;
        }

        Ok(())
    }
}

/// Iterator for iterating over MemoryData
pub struct MemoryDataIter<'a> {
    memory_data: &'a MemoryData,
    index: usize,
}

/// The item returned from the iterator
pub struct MemoryDataItem<'a> {
    address: u64,
    bytes: &'a [u8],
    access: MemoryAccess,
}

// Implement `Iterator` for the iterator struct
impl<'a> Iterator for MemoryDataIter<'a> {
    type Item = MemoryDataItem<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        let MemoryData {
            addresses,
            bytes,
            accesses,
        } = self.memory_data;

        if self.index < addresses.len() && !matches!(accesses[self.index], MemoryAccess::None) {
            let item = MemoryDataItem {
                address: addresses[self.index],
                bytes: bytes[self.index].as_slice(),
                access: accesses[self.index],
            };

            self.index += 1;
            Some(item)
        } else {
            None
        }
    }
}

// Implement a method to create an iterator from `MemoryData`
impl MemoryData {
    pub fn iter(&self) -> MemoryDataIter<'_> {
        MemoryDataIter {
            memory_data: self,
            index: 0,
        }
    }

    // Clear the contents
    pub fn clear(&mut self) {
        self.addresses.clear();
        self.bytes.clear();
        self.accesses.clear();
    }
}

#[derive(Serialize, Deserialize, Default, Clone, PartialEq, Eq)]
pub struct Debugger {
    /// The current index in the trace
    pub index: InstrIndex,

    /// The number of total entries in this trace
    pub entries: InstrIndex,

    /// The memory backing for the debugger
    pub memory: Memory,

    /// The memory diff for this step
    pub memory_diffs: Vec<MemoryData>,

    /// Memory snapshots used to restore quickly jump between
    /// instruction indexes
    memory_snapshots: BTreeMap<InstrIndex, Memory>,

    /// The distance between each memory snapshot
    memory_snapshot_delta: InstrIndex,

    /// The instruction index of the register writes
    pub register_writes: [Vec<InstrIndex>; Register::count()],

    /// The value of the register when it was last written
    pub register_values: [Vec<u64>; Register::count()],

    /// The memory reads of each address found in the trace
    pub memory_reads: BTreeMap<u64, Vec<InstrIndex>>,

    /// The memory writes of each address found in the trace
    pub memory_writes: BTreeMap<u64, Vec<InstrIndex>>,

    /// The sequence of bytes written to each address
    pub byte_history: BTreeMap<u64, (Vec<InstrIndex>, Vec<u8>)>,

    /// Pre-allocated vecs ready for use for instr index
    pub ready_vec_instr_index: Vec<Vec<InstrIndex>>,

    /// Pre-allocated vecs ready for use for bytes
    pub ready_vec_bytes: Vec<Vec<u8>>,
}

impl Debugger {
    pub fn from_file(orig_input: &Path) -> Result<Self, Error> {
        timeloop::start_profiler!();
        timeloop::scoped_timer!(Timer::ParseTrace);

        // Check if this file has already been processed into a ttdbg
        let mut input = orig_input.to_path_buf();
        let outfile = format!("{}.ttdbg.gz", input.to_string_lossy().into_owned());
        let outfile = Path::new(&outfile);
        if outfile.exists() {
            log::info!("ttdbg detected.. using ttdbg.gz");
            input = outfile.to_path_buf();
        }

        if input.extension().unwrap() == "gz" {
            log::info!("Loading from ttdbg");

            // If restoring the debugger ttdbg, then return the result
            if let Ok(mut res) = Debugger::restore(&input) {
                res.size();
                res.preallocate();
                return Ok(res);
            }

            // Restoring failed, resort to rebuilding the debugger
            input = orig_input.to_path_buf();
            log::info!("..Failed to load from ttdbg, returning to parsing the raw trace {input:?}");
        }

        log::info!("Input: {input:?}");
        let mut reader = BufReader::new(File::open(input).unwrap());

        let mut dbg = Debugger::default();
        let start = std::time::Instant::now();
        let mut timer = std::time::Instant::now();
        let mut line = String::new();
        let mut diff = TraceDiff::<REGISTER_ENTRIES>::default();
        let mut i = 0;

        timeloop::scoped_timer!(Timer::AddDiffs);
        loop {
            line.clear();
            diff.clear();

            let bytes_read = reader.read_line(&mut line).map_err(Error::Io)?;
            if bytes_read == 0 {
                break;
            }

            if timer.elapsed() > std::time::Duration::from_secs(1) {
                let virt_mem_bytes = get_virtual_mem();

                log::info!(
                    "Line: {i} | {:.2} M lines/sec | {:.2} bytes / line",
                    (i as f64 / start.elapsed().as_secs_f64()) / 1000. / 1000.,
                    virt_mem_bytes as f64 / i as f64,
                    // dbg.size() as f64 / 1024. / 1024. / 1024.
                );

                timer = std::time::Instant::now();
                // dbg.print_stats();
            }

            i += 1;

            'next_item: for item in line.split(',') {
                // Parse all of the register entries
                if &item[..1] == "r" || &item[..1] == "e" {
                    for (prefix, register) in [
                        ("rax=", Register::Rax),
                        ("eax=", Register::Eax),
                        ("rbx=", Register::Rbx),
                        ("ebx=", Register::Ebx),
                        ("rcx=", Register::Rcx),
                        ("ecx=", Register::Ecx),
                        ("rdx=", Register::Rdx),
                        ("edx=", Register::Edx),
                        ("rdi=", Register::Rdi),
                        ("edi=", Register::Edi),
                        ("rsi=", Register::Rsi),
                        ("esi=", Register::Esi),
                        ("rsp=", Register::Rsp),
                        ("esp=", Register::Esp),
                        ("rbp=", Register::Rbp),
                        ("ebp=", Register::Ebp),
                        ("rip=", Register::Rip),
                        ("eip=", Register::Eip),
                        ("r8=", Register::R8),
                        ("r9=", Register::R9),
                        ("r10=", Register::R10),
                        ("r11=", Register::R11),
                        ("r12=", Register::R12),
                        ("r13=", Register::R13),
                        ("r14=", Register::R14),
                        ("r15=", Register::R15),
                    ] {
                        if &item[..prefix.len()] == prefix {
                            let value = &item[prefix.len()..];
                            let value = parse_hex_string(value)?;
                            diff.set_register(register, value);
                            continue 'next_item;
                        }
                    }
                }

                // Parse for a memory operation next
                for (prefix, memory) in [("mr=", MemoryAccess::Read), ("mw=", MemoryAccess::Write)]
                {
                    if let Some(mem_value) = item.strip_prefix(prefix) {
                        let mut mem = mem_value.split(':');
                        let Some(addr) = mem.next() else {
                            return Err(Error::InvalidMemoryValue(mem_value.to_string()));
                        };
                        let Some(value) = mem.next() else {
                            return Err(Error::InvalidMemoryValue(mem_value.to_string()));
                        };

                        let addr = parse_hex_string(addr)?;

                        let num_bytes = value.len() / 2;
                        let mut new_value = [0; 8];
                        for i in 0..8 {
                            let range = i * 2..i * 2 + 2;
                            new_value[i] =
                                u8::from_str_radix(value.get(range).unwrap_or_else(|| "00"), 16)
                                    .unwrap();
                        }

                        // Add the parsed memory to the diff
                        diff.add_memory(addr, num_bytes as u8, new_value, memory)?;
                        continue 'next_item;
                    }
                }

                panic!("unknown item: {item}");
            }

            dbg.add_diff(&diff);
            dbg.entries += 1;
        }

        dbg.memory = Memory::default();

        {
            timeloop::scoped_timer!(Timer::TakeMemorySnapshots);

            log::info!("Taking memory snapshot");

            let start = std::time::Instant::now();

            // The number of memory snapshots to keep in memory
            let divisions = 5;

            let delta = dbg.entries / divisions;
            dbg.memory_snapshot_delta = delta;

            // Calculate the locations to take each snapshot
            let mut snapshot_locations = (0..divisions + 1)
                .map(|index| (delta * index).min(dbg.entries - 1))
                .collect::<Vec<_>>();

            // The last entry is always the end of the trace
            *snapshot_locations.iter_mut().last().unwrap() = dbg.entries - 1;

            // Take the snapshots at the locations found
            let mut memory_snapshots = BTreeMap::new();

            for entry in snapshot_locations {
                log::info!("Taking snapshot at {entry}");
                dbg.goto_index(entry);
                memory_snapshots.insert(entry, dbg.memory.clone());
            }

            // Keep a copy of the memory snapshots
            dbg.memory_snapshots = memory_snapshots;
            log::info!("Initialing memory took {:?}", start.elapsed());
        }

        // Reset the debugger for use
        dbg.reset();

        // Print the size of each element
        dbg.size();

        // Write this debugger to disk
        dbg.save(&outfile);

        // Pre-allocate
        dbg.preallocate();

        Ok(dbg)
    }

    /// Pre-allocate a bunch of vectors for use in the debugger
    pub fn preallocate(&mut self) {
        let start = std::time::Instant::now();
        for _ in 0..10000 {
            self.ready_vec_bytes.push(Vec::with_capacity(1024));
            self.ready_vec_instr_index.push(Vec::with_capacity(1024));
        }
        println!("Pre alloc took {:?}", start.elapsed());
    }

    /// Save this debugger to disk
    pub fn save(&self, filename: &Path) {
        timeloop::scoped_timer!(Timer::SaveDebuggerToDisk);
        assert!(self.ready_vec_instr_index.is_empty());
        assert!(self.ready_vec_bytes.is_empty());

        // Write the serialized data to disk
        log::info!("Writing initialized debugger to disk: {filename:?}");

        // Serialize the debugger using bincode
        log::info!("Serializing debugger");
        let start = std::time::Instant::now();
        let serialized = bincode::serialize(&self).unwrap();
        log::info!("..took {:?}", start.elapsed());

        // Compress the serialized bincode using parallel compression
        log::info!("Compressing serialized debugger");
        let start = std::time::Instant::now();
        let file = File::create(filename).unwrap();
        let mut parz = ZBuilder::<Gzip, _>::new().num_threads(0).from_writer(file);
        parz.write_all(&serialized).unwrap();
        parz.finish().unwrap();
        log::info!("..took {:?}", start.elapsed());
    }

    pub fn size(&self) -> usize {
        let Self {
            index,
            entries,
            register_writes,
            register_values,
            memory,
            memory_diffs,
            memory_snapshots,
            memory_snapshot_delta,
            memory_reads,
            memory_writes,
            byte_history,
            ready_vec_instr_index,
            ready_vec_bytes,
        } = self;

        let mut total = 0;
        let mut results = Vec::new();

        macro_rules! serialize {
            ($item:expr) => {
                let serialized = bincode::serialize(&$item).unwrap();
                let len = serialized.len();
                results.push((len, stringify!($item)));
                total += len;
            };
        }

        serialize!(index);
        serialize!(entries);
        serialize!(register_writes);
        serialize!(register_values);
        serialize!(memory);
        serialize!(memory_diffs);
        serialize!(memory_snapshots);
        serialize!(memory_snapshot_delta);
        serialize!(memory_reads);
        serialize!(memory_writes);
        serialize!(byte_history);

        results.sort();

        for (curr_len, name) in results {
            println!(
                "{name:30} {:6.2} MB {:.2}",
                curr_len as f64 / 1024. / 1024.,
                curr_len as f64 / total as f64 * 100.
            );
        }

        total
    }

    /// Restore this debugger state from disk
    pub fn restore(filename: &Path) -> Result<Self, Error> {
        timeloop::scoped_timer!(Timer::RestoreDebuggerFromDisk);

        log::info!("Initialized debugger from disk: {filename:?}");

        let mut f = BufReader::new(File::open(filename).unwrap());
        let mut data = Vec::new();
        f.read_to_end(&mut data).map_err(Error::Io)?;

        let mut bytes = Vec::new();
        let mut gz = flate2::write::GzDecoder::new(bytes);
        gz.write_all(&data).map_err(Error::Io)?;
        gz.try_finish().unwrap();
        bytes = gz.finish().unwrap();

        bincode::deserialize(&bytes[..]).map_err(Error::Bincode)
    }

    // Add the trace diff to the current debugger
    pub fn add_diff<const REGISTER_ENTRIES: usize>(&mut self, diff: &TraceDiff<REGISTER_ENTRIES>) {
        timeloop::scoped_timer!(Timer::TraceAddDiff);

        self._add_registers_from_diff(&diff);
        self._add_memory_from_diff(&diff);
    }

    fn _add_registers_from_diff<const REGISTER_ENTRIES: usize>(
        &mut self,
        diff: &TraceDiff<REGISTER_ENTRIES>,
    ) {
        timeloop::scoped_timer!(Timer::TraceAddRegisterDiff);

        let curr_index = self.entries;

        // Add the register states to the trace
        for (index, reg) in diff.register.iter().enumerate() {
            if matches!(reg, Register::None) {
                break;
            }

            let reg_val = diff.register_data[index];

            let expanded_reg = reg.expanded() as usize;
            let reg_mask = reg.bit_mask();

            let value = if reg_mask > 0 {
                let prev_val = self.register_values[expanded_reg].last().unwrap_or(&0);
                (prev_val & reg_mask) | reg_val
            } else {
                reg_val
            };

            // Add the register value if this register was written
            self.register_writes[expanded_reg].push(curr_index);
            self.register_values[expanded_reg].push(value);
        }
    }

    fn _add_memory_from_diff<const REGISTER_ENTRIES: usize>(
        &mut self,
        diff: &TraceDiff<REGISTER_ENTRIES>,
    ) {
        timeloop::scoped_timer!(Timer::TraceAddMemoryDiff);
        self.memory_diffs.push(diff.memory_data.clone());
    }

    /// Get the value of the given register at the given index. Returns the current value
    /// and whether this register was written at the given index.
    pub fn get_register_at(&self, reg: Register, index: InstrIndex) -> (u64, bool) {
        match self.register_writes[reg as usize].binary_search(&index) {
            Ok(index) => {
                // Register was written at this index, use the value there
                (self.register_values[reg as usize][index], true)
            }
            Err(0) => {
                // First time this register was written
                (0, false)
            }
            Err(index) => {
                // Register was written before
                (self.register_values[reg as usize][index - 1], false)
            }
        }
    }

    pub fn context_at(&mut self, index: InstrIndex) {
        timeloop::scoped_timer!(Timer::ContextAt);

        println!("index: {index}");

        macro_rules! get_reg_with_color {
            ($reg:ident) => {{
                let (val, was_written) = self.get_register_at(Register::$reg, index);
                let reg_name = stringify!($reg).to_ascii_uppercase();
                if was_written && Register::$reg != Register::Rip {
                    format!("{reg_name:>3}: {val:#018x}").red().to_string()
                } else {
                    format!("{reg_name:>3}: {val:#018x}").to_string()
                }
            }};
        }

        let rax = get_reg_with_color!(Rax);
        let rbx = get_reg_with_color!(Rbx);
        let rcx = get_reg_with_color!(Rcx);
        let rdx = get_reg_with_color!(Rdx);
        println!("{rax} {rbx} {rcx} {rdx}");

        let rsi = get_reg_with_color!(Rsi);
        let rdi = get_reg_with_color!(Rdi);
        let r8 = get_reg_with_color!(R8);
        let r9 = get_reg_with_color!(R9);
        println!("{rsi} {rdi} {r8} {r9}");

        let r10 = get_reg_with_color!(R10);
        let r11 = get_reg_with_color!(R11);
        let r12 = get_reg_with_color!(R12);
        let r13 = get_reg_with_color!(R13);
        println!("{r10} {r11} {r12} {r13}");

        let r14 = get_reg_with_color!(R14);
        let r15 = get_reg_with_color!(R15);
        let rsp = get_reg_with_color!(Rsp);
        let rbp = get_reg_with_color!(Rbp);
        println!("{r14} {r15} {rsp} {rbp}");

        let rip = get_reg_with_color!(Rip);
        println!("{rip}");
    }

    /// Step forward one step
    pub fn step_forward(&mut self) {
        timeloop::scoped_timer!(Timer::StepForward);

        let next_index = (self.index + 1).min(self.entries - 1);
        self.goto_index(next_index);
    }

    /// Step backward one step
    pub fn step_backward(&mut self) {
        timeloop::scoped_timer!(Timer::StepBackward);

        let prev_index = self.index.saturating_sub(1);
        self.goto_index(prev_index);
    }

    /// Reset the debugger's state
    pub fn reset(&mut self) {
        timeloop::scoped_timer!(Timer::Reset);

        self.index = 0;
        self.memory.soft_reset();
    }

    pub fn exec_command(&mut self, command: &str) -> Result<(), Error> {
        timeloop::scoped_timer!(Timer::ExecCommand);
        // log::info!("Executing command: {command}");

        let mut args = command.trim().split(' ');
        let command = args.next().unwrap();
        let arg1 = args.next();
        let arg2 = args.next();

        match command {
            "g" => {
                let index = match arg1.ok_or(Error::MissingCommandArgument)? {
                    "end" => self.entries - 1,
                    x => x.parse::<InstrIndex>().unwrap(),
                };

                log::info!("GOTO {index}");
                self.goto_index(index);
            }
            "memreads" => {
                let arg1 = arg1.ok_or(Error::MissingCommandArgument)?;
                let arg1 = arg1.replace("0x", "");
                let address = u64::from_str_radix(&arg1, 16).unwrap();

                let reads = self.get_address_reads(address);
                println!("MEM READS {address:#x} | {reads:?}");
            }
            "memwrites" => {
                let arg1 = arg1.ok_or(Error::MissingCommandArgument)?;
                let arg1 = arg1.replace("0x", "");
                let address = u64::from_str_radix(&arg1, 16).unwrap();

                let writes = self.get_address_writes(address);
                println!("MEM WRITES {address:#x} | {writes:?}");
            }
            "hexdump" => {
                let arg1 = arg1.ok_or(Error::MissingCommandArgument)?;
                let arg1 = arg1.replace("0x", "");
                let address = u64::from_str_radix(&arg1, 16).unwrap();

                let arg2 = arg2.ok_or(Error::MissingCommandArgument)?;
                let arg2 = arg2.replace("0x", "");
                let n = usize::from_str_radix(&arg2, 16).unwrap();

                self.hexdump(address, n);
            }
            "c" => {}
            "next" | "n" => self.step_forward(),
            "sb" => self.step_backward(),
            "stats" => self.print_stats(),
            "q" => return Err(Error::Quit),
            x => println!("Unknown command: {x}"),
        }

        // log::info!("Command took {:?}", start.elapsed());

        Ok(())
    }

    pub fn print_stats(&self) {
        timeloop::print!();
    }

    /// Print the context of the current location in the debugger
    pub fn print_context(&mut self) {
        timeloop::scoped_timer!(Timer::PrintContext);

        self.context_at(self.index);
    }

    /// Print a hexdump at the given address of n bytes
    pub fn hexdump(&mut self, address: u64, n: usize) {
        timeloop::scoped_timer!(Timer::Hexdump);
        self.memory.hexdump(address, n);
    }

    pub fn goto_index(&mut self, new_index: InstrIndex) {
        timeloop::scoped_timer!(Timer::GotoIndex);

        let distance = (self.index as isize - new_index as isize).abs() as InstrIndex;

        // Check if we are moving more than the current memory snapshot delta
        // If so, restore the memory to the closest snapshot, and then move
        // the index
        let delta = self.memory_snapshot_delta;
        if distance > delta && !self.memory_snapshots.is_empty() {
            let start = std::time::Instant::now();

            let mut which_snapshot = self.index + delta;
            let mut best_distance = u64::MAX;

            // Figure out the closest snapshot location
            for addr in self.memory_snapshots.keys() {
                let curr_distance = (new_index as isize - *addr as isize).unsigned_abs() as u64;
                if curr_distance < best_distance {
                    best_distance = curr_distance;
                    which_snapshot = *addr;
                }
            }

            log::debug!(
                "{} -> {}: Which snapshot {which_snapshot} {:?}",
                self.index,
                new_index,
                self.memory_snapshots.keys()
            );

            self.memory = self.memory_snapshots.get(&which_snapshot).unwrap().clone();

            log::debug!("Cloning final memory took: {:?}", start.elapsed());

            self.index = which_snapshot;
        }

        self.update_memory_to(new_index);
        self.index = new_index;
    }

    /// Update the memory from the current index
    fn update_memory_to(&mut self, target_index: InstrIndex) {
        timeloop::scoped_timer!(Timer::UpdateMemory);

        if target_index >= self.index {
            timeloop::scoped_timer!(Timer::UpdateMemoryForward);
            // Update forward (easy path)
            for (i, diffs) in self.memory_diffs[self.index as usize..target_index as usize + 1]
                .iter()
                .enumerate()
            {
                for MemoryDataItem {
                    address,
                    bytes,
                    access,
                } in diffs.iter()
                {
                    self.memory.set_bytes(
                        self.index + i as InstrIndex,
                        address,
                        bytes,
                        access,
                        &mut self.memory_reads,
                        &mut self.memory_writes,
                        &mut self.byte_history,
                        &mut self.ready_vec_bytes,
                        &mut self.ready_vec_instr_index,
                    );
                }
            }
        } else {
            timeloop::scoped_timer!(Timer::UpdateMemoryBackward);

            // Update backward (hard-er path)
            let mut writes = BTreeMap::new();

            {
                timeloop::scoped_timer!(Timer::UpdateMemoryBackwardCollapseWrites);

                // Collapse all of the writes to the memory addresses within the backward range
                // to only one write per byte
                for (i, diffs) in self.memory_diffs[target_index as usize..self.index as usize + 1]
                    .iter()
                    .enumerate()
                    .rev()
                {
                    for MemoryDataItem {
                        address,
                        bytes,
                        access,
                    } in diffs.iter()
                    {
                        if matches!(access, MemoryAccess::Write)
                            || matches!(access, MemoryAccess::Read)
                        {
                            for offset in 0..bytes.len() as u64 {
                                writes.insert(address + offset, i as InstrIndex + target_index);
                            }
                        }
                    }
                }
            }

            // Printing the current byte history
            /*
            for (addr, indexes) in self.memory.byte_history.iter() {
                println!("{addr:#x} {indexes:?}");
            }
            */

            for (addr, last_write) in writes.iter() {
                let Some((addresses, bytes)) = self.byte_history.get_mut(addr) else {
                    panic!("Invalid byte history state");
                };

                let result = addresses.binary_search(last_write);

                log::debug!(
                    "XXX {}..{} {addr:#x} {last_write} -> {result:?}",
                    self.index,
                    target_index
                );

                match result {
                    Err(0) => {
                        // If there was no prior byte in this address's history.
                        self.memory
                            .set_byte_state(*addr, memory::ByteState::Unknown);
                    }
                    Err(index) => {
                        // Get the last known byte before this address was written
                        let last_byte = bytes[index - 1];
                        log::debug!("  --> {last_write} ({last_byte:#x})");

                        self.memory.set_byte_state(*addr, memory::ByteState::Known);
                        self.memory.set_byte(*addr, last_byte);
                    }
                    Ok(mut index) => {
                        log::debug!("Found instr: {} Target: {}", addresses[index], target_index);
                        log::debug!("Addresses {addresses:?}");
                        while index > 0 && addresses[index] > target_index {
                            log::debug!("  --> SUB INDEX");
                            index = index.saturating_sub(1);
                            log::debug!(
                                "  --> Found instr: {} Target: {} Byte: {:#x}",
                                addresses[index],
                                target_index,
                                bytes[index]
                            );
                        }

                        // If the found index where this address is written is
                        // AFTER the current instruction index, then this byte
                        // is unknown
                        if target_index < addresses[index] {
                            log::debug!("  --> UNSET {:#x}", *addr);
                            self.memory
                                .set_byte_state(*addr, memory::ByteState::Unknown);
                            let bytes = self.memory.read(*addr, 4);
                            log::debug!("  --> {:#x} UNSET {:x?}", *addr, bytes);
                        } else {
                            // Value was found, use this index as the last byte in the history
                            let last_byte = bytes[index];
                            log::debug!("  --> {last_write} ({last_byte:#x})");

                            self.memory.set_byte_state(*addr, memory::ByteState::Known);
                            self.memory.set_byte(*addr, last_byte);
                        }
                    }
                }
            }
        }
    }

    /// Get the instruction indexes of the reads of the given address
    pub fn get_address_reads(&self, address: u64) -> Vec<InstrIndex> {
        timeloop::scoped_timer!(Timer::GetAddressReads);

        self.memory_reads
            .get(&address)
            .cloned()
            .unwrap_or_else(|| Vec::new())
    }

    /// Get the instruction indexes of the writes of the given address
    pub fn get_address_writes(&self, address: u64) -> Vec<InstrIndex> {
        timeloop::scoped_timer!(Timer::GetAddressWrites);

        self.memory_writes
            .get(&address)
            .cloned()
            .unwrap_or_else(|| Vec::new())
    }

    /// Get the instruction indexes of the reads and writes of the given address
    pub fn get_address_access(&self, address: u64) -> Vec<InstrIndex> {
        timeloop::scoped_timer!(Timer::GetAddressAccesses);

        let mut result = BTreeSet::new();

        // Add the reads
        let reads = self.get_address_reads(address);
        for read in reads {
            result.insert(read);
        }

        // Add the writes
        let writes = self.get_address_writes(address);
        for write in writes {
            result.insert(write);
        }

        // Sort the unique results
        let mut result = result.iter().cloned().collect::<Vec<_>>();
        result.sort();
        result
    }
}

pub enum CommandResult {
    Continue,
    Quit,
}

#[derive(Clone)]
pub struct TraceDiff<const REGISTER_ENTRIES: usize> {
    pub register: [Register; REGISTER_ENTRIES],
    pub register_data: [u64; REGISTER_ENTRIES],
    pub memory_data: MemoryData,
}

impl<const REGISTER_ENTRIES: usize> TraceDiff<REGISTER_ENTRIES> {
    pub fn set_register(&mut self, new_reg: Register, new_value: u64) {
        timeloop::scoped_timer!(Timer::SetRegister);

        for (index, reg) in self.register.iter_mut().enumerate() {
            if !matches!(reg, Register::None) {
                continue;
            }

            self.register_data[index] = new_value;
            *reg = new_reg;
            return;
        }

        panic!("Too many register entries. Increase REGISTER_ENTRIES");
    }

    pub fn add_memory(
        &mut self,
        address: u64,
        num_bytes: u8,
        new_bytes: [u8; 8],
        access: MemoryAccess,
    ) -> Result<(), Error> {
        timeloop::scoped_timer!(Timer::AddMemory);

        self.memory_data.addresses.push(address);
        self.memory_data
            .bytes
            .push(new_bytes[..num_bytes as usize].to_vec());
        self.memory_data.accesses.push(access);

        Ok(())
    }

    // Clear the contents
    pub fn clear(&mut self) {
        self.register = [Register::None; REGISTER_ENTRIES];
        self.register_data = [0; REGISTER_ENTRIES];
        self.memory_data.clear();
    }
}

impl<const REGISTER_ENTRIES: usize> std::default::Default for TraceDiff<REGISTER_ENTRIES> {
    fn default() -> Self {
        TraceDiff {
            register: [Register::None; REGISTER_ENTRIES],
            register_data: [0; REGISTER_ENTRIES],
            memory_data: MemoryData::default(),
        }
    }
}

impl<const REGISTER_ENTRIES: usize> std::fmt::Debug for TraceDiff<REGISTER_ENTRIES> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::result::Result<(), std::fmt::Error> {
        for (reg, val) in self.register.iter().zip(self.register_data.iter()) {
            if matches!(reg, Register::None) {
                break;
            }

            write!(f, "{reg:?}:{val:#x} ")?;
        }

        write!(f, "{:?}", self.memory_data)
    }
}

pub fn start_profiler() {
    timeloop::start_profiler!();
}

fn get_virtual_mem() -> u64 {
    let status_reader = BufReader::new(File::open("/proc/self/status").unwrap());
    for line in status_reader.lines() {
        let line = line.unwrap();

        if line.contains("VmSize") {
            let line = line.replace("  ", " ");
            let x = line
                .split(' ')
                .nth(1)
                .unwrap_or_else(|| "0")
                .parse::<u64>()
                .unwrap_or(0);

            return x * 1024;
        }
    }

    0
}
