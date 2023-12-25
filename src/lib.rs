#![feature(thread_id_value)]

mod colors;
use colors::Colorized;

pub mod memory;
mod utils;

use gzp::{deflate::Gzip, ZBuilder};

use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufRead, BufReader, Read, Write};
use std::sync::Arc;

use memory::Memory;
use utils::parse_hex_string;

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

/// The maximum number of register entries allowed per trace entry line
/// before heap allocating storage for the register diff
const REGISTER_ENTRIES: usize = 10;

pub type InstrIndex = u32;

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
        Memory_WriteBytes,

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
    ParseIntError(String),
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

    pub fn from_str(input: &str) -> Option<Self> {
        match input {
            "rax" => Some(Register::Rax),
            "rbx" => Some(Register::Rbx),
            "rcx" => Some(Register::Rcx),
            "rdx" => Some(Register::Rdx),
            "rsi" => Some(Register::Rsi),
            "rdi" => Some(Register::Rdi),
            "rip" => Some(Register::Rip),
            "rsp" => Some(Register::Rsp),
            "rbp" => Some(Register::Rbp),
            "r8" => Some(Register::R8),
            "r9" => Some(Register::R9),
            "r10" => Some(Register::R10),
            "r11" => Some(Register::R11),
            "r12" => Some(Register::R12),
            "r13" => Some(Register::R13),
            "r14" => Some(Register::R14),
            "r15" => Some(Register::R15),

            "eax" => Some(Register::Eax),
            "ebx" => Some(Register::Ebx),
            "ecx" => Some(Register::Ecx),
            "edx" => Some(Register::Edx),
            "esi" => Some(Register::Esi),
            "edi" => Some(Register::Edi),
            "dx" => Some(Register::Dx),
            "dl" => Some(Register::Dl),
            "eip" => Some(Register::Eip),
            "esp" => Some(Register::Esp),
            "ebp" => Some(Register::Ebp),
            _ => None,
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
    ReadWrite,
}

/// The set of memory data diffs per line in the trace
#[derive(Serialize, Deserialize, Clone, PartialEq, Eq)]
pub struct MemoryData {
    addresses: Vec<u64>,
    bytes: Vec<Vec<u8>>,
    accesses: Vec<MemoryAccess>,
    num_bytes: Vec<u8>,
}

impl std::default::Default for MemoryData {
    fn default() -> Self {
        Self {
            addresses: Vec::with_capacity(8),
            bytes: Vec::with_capacity(8),
            accesses: Vec::with_capacity(8),
            num_bytes: Vec::with_capacity(8),
        }
    }
}

impl std::fmt::Debug for MemoryData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::result::Result<(), std::fmt::Error> {
        let MemoryData {
            addresses,
            bytes,
            accesses,
            num_bytes,
        } = self;
        for (index, access) in accesses.iter().enumerate() {
            let address = addresses[index];
            let curr_bytes = &bytes[index];
            let curr_num_bytes = num_bytes[index];

            let operation = match access {
                MemoryAccess::Write => "<-",
                MemoryAccess::Read => "->",
                _ => unreachable!(),
            };

            write!(f, "{access:?} {address:#x} {operation} {curr_num_bytes}")?;

            for byte in curr_bytes.iter().take(curr_num_bytes.into()) {
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
    num_bytes: u8,
}

// Implement `Iterator` for the iterator struct
impl<'a> Iterator for MemoryDataIter<'a> {
    type Item = MemoryDataItem<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        let MemoryData {
            addresses,
            bytes,
            accesses,
            num_bytes,
        } = self.memory_data;

        if self.index < addresses.len() && !matches!(accesses[self.index], MemoryAccess::None) {
            let item = MemoryDataItem {
                address: addresses[self.index],
                bytes: bytes[self.index].as_slice(),
                access: accesses[self.index],
                num_bytes: num_bytes[self.index],
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

    /// Add a new element into the memory data
    pub fn add(&mut self, address: u64, bytes: Vec<u8>, access: MemoryAccess, num_bytes: u8) {
        self.addresses.push(address);
        self.bytes.push(bytes);
        self.accesses.push(access);
        self.num_bytes.push(num_bytes);
    }

    // Clear the contents of this memory data
    pub fn clear(&mut self) {
        let Self {
            addresses,
            bytes,
            accesses,
            num_bytes,
        } = self;

        addresses.clear();
        bytes.clear();
        accesses.clear();
        num_bytes.clear();
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

    /// The memory diff for each step
    pub memory_diffs: Vec<MemoryData>,

    /// Memory snapshots used to restore quickly jump between
    /// instruction indexes
    memory_snapshots: BTreeMap<InstrIndex, Memory>,

    /// The distance between each memory snapshot
    pub memory_snapshot_delta: InstrIndex,

    /// The instruction index of the register writes
    pub register_writes: [Vec<InstrIndex>; Register::count()],

    /// The value of the register when it was last written
    pub register_values: [Vec<u64>; Register::count()],

    /// The memory reads of each address found in the trace
    pub memory_reads: BTreeMap<u64, Vec<InstrIndex>>,

    /// The memory writes of each address found in the trace
    pub memory_writes: BTreeMap<u64, Vec<InstrIndex>>,

    /// The addresses executed in the trace
    pub rips: BTreeMap<u64, Vec<InstrIndex>>,

    /// The sequence of bytes written to each address
    pub byte_history: BTreeMap<u64, (Vec<InstrIndex>, Vec<u8>)>,

    /// Pre-allocated vecs ready for use for instr index
    pub ready_vec_instr_index: Vec<Vec<InstrIndex>>,

    /// Pre-allocated vecs ready for use for bytes
    pub ready_vec_bytes: Vec<Vec<u8>>,

    /// Has the debugger caches been initialized
    pub caches_initialized: bool,
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
                // res.size();
                res.preallocate();
                return Ok(res);
            }

            // Restoring failed, resort to rebuilding the debugger
            input = orig_input.to_path_buf();
            log::info!("..Failed to load from ttdbg, returning to parsing the raw trace {input:?}");
        }

        log::info!("Gathering lines");
        let start = std::time::Instant::now();
        let mut reader = BufReader::new(File::open(&input).unwrap());
        let lines = reader.lines().flat_map(|x| x).collect::<Vec<_>>();
        let num_lines = lines.len();
        log::info!("Input Lines: {num_lines}");
        log::info!("READ: {:?}", start.elapsed());

        let mut reader = BufReader::new(File::open(&input).unwrap());
        let mut dbg = Debugger::default();
        let start = std::time::Instant::now();
        let mut timer = std::time::Instant::now();
        let mut i = 0;
        let num_cores = 8;
        let line_delta = num_lines / num_cores;
        let lines = Arc::new(lines);

        fn parse_file_thread(
            core_id: usize,
            lines: Arc<Vec<String>>,
            start_index: usize,
            num_lines: usize,
        ) -> Result<Vec<TraceDiff<REGISTER_ENTRIES>>, Error> {
            let start = std::time::Instant::now();
            let mut timer = std::time::Instant::now();
            let mut line = String::new();
            let mut results = Vec::new();
            timeloop::scoped_timer!(Timer::AddDiffs);

            let mut diffs = (0..num_lines)
                .map(|_| TraceDiff::<REGISTER_ENTRIES>::default())
                .collect::<Vec<_>>();

            for index in start_index..start_index + num_lines {
                let line = &lines[index];

                if line.is_empty() {
                    continue;
                }

                let mut diff = diffs.pop().unwrap();

                if timer.elapsed() > std::time::Duration::from_millis(100) {
                    let virt_mem_bytes = get_virtual_mem();

                    log::info!(
                        "Line: {index}/{} | {:.2} M lines/sec | {:.2} bytes / line",
                        start_index + num_lines,
                        ((index - start_index) as f64 / start.elapsed().as_secs_f64())
                            / 1000.
                            / 1000.,
                        virt_mem_bytes as f64 / (index - start_index) as f64,
                        // dbg.size() as f64 / 1024. / 1024. / 1024.
                    );

                    timer = std::time::Instant::now();
                }

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
                    for (prefix, memory) in
                        [("mr=", MemoryAccess::Read), ("mw=", MemoryAccess::Write)]
                    {
                        if let Some(mem_value) = item.strip_prefix(prefix) {
                            let mut mem = mem_value.split(':');
                            let Some(addr) = mem.next() else {
                                return Err(Error::InvalidMemoryValue(mem_value.to_string()));
                            };

                            let Some(value) = mem.next() else {
                                return Err(Error::InvalidMemoryValue(mem_value.to_string()));
                            };

                            let mut new_value = [0; 16];
                            let addr = parse_hex_string(addr)?;
                            let num_bytes: u8;

                            if let Some(num_bytes_str) = value.strip_prefix('s') {
                                // Parse: 0xdeadbeef:s8
                                // Memory access at 0xdeadbeef for 8 bytes
                                num_bytes = num_bytes_str
                                    .parse::<u8>()
                                    .map_err(|_| Error::ParseIntError(value.to_string()))?;
                            } else {
                                // Parse: 0xdeadbeef:12345678
                                // Memory access at 0xdeadbeef for bytes [0x12, 0x34, 0x56, x78]
                                num_bytes = value.len() as u8 / 2;

                                for i in 0..num_bytes as usize {
                                    let range = i * 2..i * 2 + 2;
                                    new_value[i] = u8::from_str_radix(
                                        value.get(range).unwrap_or_else(|| "00"),
                                        16,
                                    )
                                    .unwrap();
                                }
                            }

                            // Add the parsed memory to the diff
                            diff.add_memory(addr, num_bytes, new_value, memory)?;
                            continue 'next_item;
                        }
                    }

                    panic!("unknown item: {item}");
                }

                results.push(diff);
            }

            Ok(results)
        }

        let mut threads = Vec::new();
        for core_id in 0..num_cores {
            let start_index = core_id * line_delta;
            let lines = lines.clone();

            let thread = std::thread::spawn(move || {
                parse_file_thread(core_id, lines, start_index, line_delta)
            });

            threads.push(thread);
        }

        let start = std::time::Instant::now();
        let mut core_id = 0;
        for thread in threads {
            match thread.join().unwrap() {
                Ok(diffs) => {
                    log::info!("Adding {} diffs from thread {}", diffs.len(), core_id);
                    for diff in diffs {
                        dbg.add_diff(&diff);
                        dbg.entries += 1;
                    }
                }
                Err(e) => {
                    panic!("Thread failed: {e:?}")
                }
            }

            core_id += 1;
        }

        log::info!("Loading diffs from tenet trace: {:?}", start.elapsed());

        {
            timeloop::scoped_timer!(Timer::TakeMemorySnapshots);
            log::info!("Taking memory snapshot");

            dbg.memory = Memory::default();

            let start = std::time::Instant::now();

            // The number of memory snapshots to keep in memory
            let divisions = (dbg.entries / 250_000).max(1);

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
                std::io::stdout().flush().unwrap();

                dbg.goto_index(entry);
                memory_snapshots.insert(entry, dbg.memory.clone());
            }

            // Keep a copy of the memory snapshots
            dbg.memory_snapshots = memory_snapshots;
            log::info!("Initialing memory took {:?}", start.elapsed());

            // The debugger's caches have now been initialized
            dbg.caches_initialized = true;
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
            rips,
            byte_history,
            ready_vec_instr_index,
            ready_vec_bytes,
            caches_initialized,
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
        serialize!(rips);
        serialize!(byte_history);
        serialize!(ready_vec_instr_index);
        serialize!(ready_vec_bytes);
        serialize!(caches_initialized);

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

    /// Get the value of the given register at the current index. Returns the current value
    /// and whether this register was written at the given index.
    pub fn get_register(&self, reg: Register) -> (u64, bool) {
        self.get_register_at(reg, self.index)
    }

    pub fn context_at(&mut self, index: InstrIndex) {
        timeloop::scoped_timer!(Timer::ContextAt);

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
            // Update forward
            timeloop::scoped_timer!(Timer::UpdateMemoryForward);

            // Collapse all of the memory writes for this difference
            let mut memory_writes = BTreeMap::new();
            let start = std::time::Instant::now();

            // Collapse all of the memory and register writes for the forward pass
            for (index, diffs) in self.memory_diffs[self.index as usize..target_index as usize + 1]
                .iter()
                .enumerate()
            {
                let instr_index = index as InstrIndex;
                let rip = self.register_values[Register::Rip as usize][instr_index as usize];

                // Add this instruction to the known executed instructions
                self.rips
                    .entry(rip)
                    .or_insert_with(|| self.ready_vec_instr_index.pop().unwrap_or_default())
                    .push(instr_index);

                for MemoryDataItem {
                    address,
                    bytes,
                    access,
                    num_bytes,
                } in diffs.iter()
                {
                    // If the debugger hasn't been initialized, cache all of the
                    // memory read and write locations for each instruction
                    if !self.caches_initialized {
                        // Set the requested bytes to the memory
                        for index in 0..num_bytes as usize {
                            let byte = bytes[index];
                            let curr_addr = address + index as u64;

                            // Mark this byte in the history for this instruction index
                            let (indexes, bytes) =
                                self.byte_history.entry(curr_addr).or_insert_with(|| {
                                    (
                                        self.ready_vec_instr_index.pop().unwrap_or_default(),
                                        self.ready_vec_bytes.pop().unwrap_or_default(),
                                    )
                                });

                            if *indexes.last().unwrap_or(&0) <= instr_index {
                                indexes.push(instr_index);
                                bytes.push(byte);

                                if matches!(access, MemoryAccess::Read) {
                                    self.memory_reads
                                        .entry(curr_addr)
                                        .or_insert_with(|| {
                                            self.ready_vec_instr_index.pop().unwrap_or_default()
                                        })
                                        .push(instr_index);
                                } else if matches!(access, MemoryAccess::Write) {
                                    self.memory_writes
                                        .entry(curr_addr)
                                        .or_insert_with(|| {
                                            self.ready_vec_instr_index.pop().unwrap_or_default()
                                        })
                                        .push(instr_index);
                                }
                            }
                        }
                    }

                    for (i, byte) in bytes.iter().enumerate() {
                        memory_writes.insert(address + i as u64, *byte);
                    }
                }
            }

            log::info!(
                "{:?} Done.. Entries {}",
                start.elapsed(),
                memory_writes.len()
            );
            let start = std::time::Instant::now();

            let mut memory_writes = memory_writes.iter().collect::<Vec<_>>();
            memory_writes.sort();
            let mut byte_index = 0;
            let mut temp_buf = [0_u8; 64];
            let mut prev_addr = 0;
            for (addr, byte) in memory_writes {
                if addr.saturating_sub(1) != prev_addr || byte_index >= temp_buf.len() {
                    // Found a non-consecutive address or the temp buffer is full, write the current buffer
                    let write_addr = prev_addr - byte_index as u64 + 1;
                    log::debug!(
                        "Writing out: {write_addr:#x} {:x?}",
                        &temp_buf[..byte_index]
                    );

                    // Write out these consecutive bytes
                    self.memory.write_bytes(write_addr, &temp_buf[..byte_index]);

                    // Reset the temp bufs and index
                    temp_buf = [0_u8; 64];
                    byte_index = 0;
                }

                // Add the current entry to the temp buffer
                log::debug!("Addr: {addr:#x} Byte: {byte:#x}");
                temp_buf[byte_index] = *byte;
                byte_index += 1;
                prev_addr = *addr;
            }
            log::info!("{:?} Done2..", start.elapsed());
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
                        num_bytes,
                    } in diffs.iter()
                    {
                        if matches!(access, MemoryAccess::Write)
                            || matches!(access, MemoryAccess::Read)
                        {
                            for offset in 0..num_bytes as u64 {
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
                        // self.memory.set_byte_state(*addr, memory::ByteState::Unknown);
                        self.memory.set_byte(*addr, 0);
                    }
                    Err(index) => {
                        // Get the last known byte before this address was written
                        let last_byte = bytes[index - 1];
                        log::debug!("  --> {last_write} ({last_byte:#x})");

                        // self.memory.set_byte_state(*addr, memory::ByteState::Known);
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
                            // log::debug!("  --> UNSET {:#x}", *addr);
                            //self.memory.set_byte_state(*addr, memory::ByteState::Unknown);
                            // let bytes = self.memory.read(*addr, 4);
                            // log::debug!("  --> {:#x} UNSET {:x?}", *addr, bytes);
                        } else {
                            // Value was found, use this index as the last byte in the history
                            let last_byte = bytes[index];
                            log::debug!("  --> 123 {last_write} ({last_byte:#x})");

                            // self.memory.set_byte_state(*addr, memory::ByteState::Known);
                            self.memory.set_byte(*addr, last_byte);
                        }
                    }
                }
            }
        }
    }

    /// Get the instruction indexes of the reads of the given address
    pub fn get_memory_reads(&self, address: u64) -> Vec<InstrIndex> {
        timeloop::scoped_timer!(Timer::GetAddressReads);

        self.memory_reads
            .get(&address)
            .cloned()
            .unwrap_or_else(|| Vec::new())
    }

    /// Get the instruction indexes of the writes of the given address
    pub fn get_memory_writes(&self, address: u64) -> Vec<InstrIndex> {
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
        let reads = self.get_memory_reads(address);
        for read in reads {
            result.insert(read);
        }

        // Add the writes
        let writes = self.get_memory_writes(address);
        for write in writes {
            result.insert(write);
        }

        // Sort the unique results
        let mut result = result.iter().cloned().collect::<Vec<_>>();
        result.sort();
        result
    }

    /// Go to the previous read of the given address
    pub fn goto_prev_read(&mut self, address: u64) {
        let reads = self.get_memory_reads(address);
        log::info!("reads: {reads:?}");

        if reads.is_empty() {
            log::warn!("No memory reads from {address:#x}");
            return;
        }

        let result = reads.binary_search(&self.index);

        // Get the previous index
        let next_index = match result {
            Ok(index) => index,
            Err(0) => {
                log::warn!("No previous memory reads from {address:#x}");
                return;
            }
            Err(index) => index - 1,
        };

        log::info!("{} -> {}", self.index, next_index);

        // Goto the found index
        self.goto_index(next_index as InstrIndex);
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
        new_bytes: [u8; 16],
        access: MemoryAccess,
    ) -> Result<(), Error> {
        timeloop::scoped_timer!(Timer::AddMemory);
        self.memory_data.add(
            address,
            new_bytes[..num_bytes.into()].to_vec(),
            access,
            num_bytes,
        );
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
