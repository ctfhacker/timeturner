#![feature(thread_id_value)]

// mod auto_restore_cell;
// use auto_restore_cell::AutoRestoreCell;
mod memory;
mod utils;

use memory::Memory;
use utils::parse_hex_string;

use std::collections::BTreeMap;
use std::mem::{size_of, size_of_val};
use std::path::Path;

/// The maximum number of memory entries allowed per trace entry line
/// before heap allocating storage for the memory diff
const MEMORY_ENTRIES: usize = 1;

/// The maximum number of register entries allowed per trace entry line
/// before heap allocating storage for the register diff
const REGISTER_ENTRIES: usize = 10;

timeloop::impl_enum!(
    #[derive(Debug, Copy, Clone, Eq, PartialEq)]
    pub enum Timer {
        Remaining,
        ParseInput,
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

        Memory_AllocatePage,
        Memory_SetMemory,
        Memory_GetPageIndex,
        Memory_IsStraddlingPage,
        Memory_Hexdump,
        Memory_Read,
        Memory_SetBytes,
        Memory_SoftReset,
        Memory_SetByte,
        Memory_CheckNewInstrIndex,
    }
);

timeloop::create_profiler!(Timer);

#[derive(Debug)]
pub enum Error {
    Io(std::io::Error),
    ParseIntError(std::num::ParseIntError),
    InvalidMemoryValue(String),
    AddMemory,
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
#[derive(Debug, Default, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MemoryAccess {
    #[default]
    None,
    Read,
    Write,
}

/// The address and bytes that were read or wrriten to that address
#[derive(Copy, Clone)]
pub struct MemoryDiff {
    /// The virtual address of this memory access
    address: u64,

    /// The bytes read/written from/to this access
    bytes: [u8; 8],
}

/// The set of memory data diffs per line in the trace
#[derive(Clone)]
pub enum MemoryData<const MEMORY_ENTRIES: usize> {
    Static(
        (
            [MemoryDiff; MEMORY_ENTRIES],
            [u8; MEMORY_ENTRIES],
            [MemoryAccess; MEMORY_ENTRIES],
        ),
    ),
    Dynamic((Vec<u64>, Vec<Vec<u8>>, Vec<MemoryAccess>)),
}

impl<const MEMORY_ENTRIES: usize> std::fmt::Debug for MemoryData<MEMORY_ENTRIES> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::result::Result<(), std::fmt::Error> {
        match self {
            Self::Static((values, size_of_values, accesses)) => {
                for (index, access) in accesses.iter().enumerate() {
                    if matches!(access, MemoryAccess::None) {
                        return Ok(());
                    };

                    let num_bytes = size_of_values[index] as usize;

                    let MemoryDiff { address, bytes } = values[index];
                    let operation = match access {
                        MemoryAccess::Write => "<-",
                        MemoryAccess::Read => "->",
                        _ => unreachable!(),
                    };

                    write!(f, "{access:?} {address:#x} {operation} ")?;

                    for byte in &bytes[..num_bytes] {
                        write!(f, "{:02x} ", byte)?;
                    }
                }
            }

            Self::Dynamic((addresses, bytes, accesses)) => {
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
            }
        }

        Ok(())
    }
}

impl<const MEMORY_ENTRIES: usize> std::default::Default for MemoryData<MEMORY_ENTRIES> {
    fn default() -> Self {
        Self::Static((
            [MemoryDiff {
                address: 0,
                bytes: [0; 8],
            }; MEMORY_ENTRIES],
            [0; MEMORY_ENTRIES],
            [MemoryAccess::None; MEMORY_ENTRIES],
        ))
    }
}

/// Iterator for iterating over MemoryData
pub struct MemoryDataIter<'a, const MEMORY_ENTRIES: usize> {
    memory_data: &'a MemoryData<MEMORY_ENTRIES>,
    index: usize,
}

/// The item returned from the iterator
pub struct MemoryDataItem<'a> {
    address: u64,
    bytes: &'a [u8],
    access: MemoryAccess,
}

// Implement `Iterator` for the iterator struct
impl<'a, const MEMORY_ENTRIES: usize> Iterator for MemoryDataIter<'a, MEMORY_ENTRIES> {
    type Item = MemoryDataItem<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.memory_data {
            MemoryData::Static((diffs, bytes, accesses)) => {
                if self.index < MEMORY_ENTRIES
                    && !matches!(accesses[self.index], MemoryAccess::None)
                {
                    let MemoryDiff {
                        address,
                        bytes: memory_bytes,
                    } = &diffs[self.index];

                    let size = bytes[self.index] as usize;

                    let item = MemoryDataItem {
                        address: *address,
                        bytes: &memory_bytes[..size],
                        access: accesses[self.index],
                    };

                    self.index += 1;
                    Some(item)
                } else {
                    None
                }
            }
            MemoryData::Dynamic((addresses, bytes, accesses)) => {
                if self.index < addresses.len()
                    && !matches!(accesses[self.index], MemoryAccess::None)
                {
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
    }
}

// Implement a method to create an iterator from `MemoryData`
impl<const MEMORY_ENTRIES: usize> MemoryData<MEMORY_ENTRIES> {
    pub fn iter(&self) -> MemoryDataIter<'_, MEMORY_ENTRIES> {
        MemoryDataIter {
            memory_data: self,
            index: 0,
        }
    }
}

#[derive(Default, Clone)]
pub struct Debugger {
    /// The current index in the trace
    pub index: usize,

    /// The number of total entries in this trace
    pub entries: usize,

    /// The register states of each step
    pub registers: [Vec<u64>; Register::count()],

    /// The memory backing for the debugger
    // pub memory: AutoRestoreCell<Memory>,
    pub memory: Memory,

    /// The memory diff for this step
    pub memory_diffs: Vec<MemoryData<MEMORY_ENTRIES>>,
}

impl Debugger {
    pub fn from_file(input: &Path) -> Result<Self, Error> {
        timeloop::start_profiler!();
        timeloop::scoped_timer!(Timer::ParseTrace);

        let data = timeloop::time_work!(Timer::ReadFile, {
            std::fs::read_to_string(input).map_err(Error::Io)?
        });

        let mut dbg = Debugger::default();

        for line in data.lines() {
            let mut diff = TraceDiff::<REGISTER_ENTRIES, MEMORY_ENTRIES>::default();
            'next_item: for item in line.split(',') {
                timeloop::scoped_timer!(Timer::ParseRegister);

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
                        timeloop::scoped_timer!(Timer::StripPrefix);

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
                    timeloop::scoped_timer!(Timer::ParseMemory);

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

            dbg.add_diff(diff);
        }

        // dbg.memory = AutoRestoreCell::new(Memory::default());
        dbg.memory = Memory::default();

        dbg.update_memory_to(0);

        Ok(dbg)
    }

    pub fn size(&self) -> usize {
        let Debugger {
            index: _,
            entries: _,
            registers,
            memory,
            memory_diffs,
        } = self;

        let mut size = size_of::<Debugger>();
        size += registers.iter().fold(0, |acc, reg| acc + size_of_val(reg));
        size += size_of_val(&memory);
        size += memory_diffs
            .iter()
            .fold(0, |acc, diff| acc + size_of_val(diff));

        size
    }

    pub fn add_diff<const REGISTER_ENTRIES: usize>(
        &mut self,
        diff: TraceDiff<REGISTER_ENTRIES, MEMORY_ENTRIES>,
    ) {
        timeloop::scoped_timer!(Timer::TraceAddDiff);

        self._add_registers_from_diff(&diff);
        self._add_memory_from_diff(&diff);
    }

    fn _add_registers_from_diff<const REGISTER_ENTRIES: usize>(
        &mut self,
        diff: &TraceDiff<REGISTER_ENTRIES, MEMORY_ENTRIES>,
    ) {
        timeloop::scoped_timer!(Timer::TraceAddRegisterDiff);

        let mut seen_registers = [false; Register::count()];

        // Add the register states to the trace
        for (index, reg) in diff.register.iter().enumerate() {
            if matches!(reg, Register::None) {
                break;
            }

            let reg_val = diff.register_data[index];

            let expanded_reg = reg.expanded() as usize;
            let reg_mask = reg.bit_mask();

            let value = if reg_mask > 0 {
                let prev_val = self.registers[expanded_reg].last().unwrap_or(&0);
                (prev_val & reg_mask) | reg_val
            } else {
                reg_val
            };

            self.registers[expanded_reg].push(value);
            seen_registers[expanded_reg] = true;
        }

        self.entries += 1;

        // For registers that were not changed this diff, use the previous value of the register
        for (index, reg) in seen_registers.iter().enumerate() {
            if !reg {
                let old_val = *self.registers[index].last().unwrap_or(&0);
                self.registers[index].push(old_val);
            }
        }

        // Ensure all data in this struct of arrays were updated
        let mut windows = self.registers.windows(2);
        while let Some([a, b]) = windows.next() {
            assert!(a.len() == b.len());
        }
    }

    fn _add_memory_from_diff<const REGISTER_ENTRIES: usize>(
        &mut self,
        diff: &TraceDiff<REGISTER_ENTRIES, MEMORY_ENTRIES>,
    ) {
        timeloop::scoped_timer!(Timer::TraceAddMemoryDiff);
        self.memory_diffs.push(diff.memory_data.clone());

        /*
        for (
            index,
            MemoryDataItem {
                address,
                bytes,
                access,
            },
        ) in self.memory_diffs[self.index].iter().enumerate()
        {
            /*
            match access {
                MemoryAccess::Read => {
                    self.memory
                        .memory_reads
                        .entry(address)
                        .or_default()
                        .push(index);
                }
                MemoryAccess::Write => {
                    self.memory
                        .memory_writes
                        .entry(address)
                        .or_default()
                        .push(index);
                }
                MemoryAccess::None => {
                    // Nothing to add for none memory access
                }
            }
            */

            self.memory.set_bytes(address, bytes);
        }
        */
    }

    pub fn context_at(&mut self, index: usize) {
        timeloop::scoped_timer!(Timer::ContextAt);

        println!("index: {index}");
        println!("Total size: {:.2} MB", self.size() as f64 / 1024. / 1024.);
        println!(
            "RAX: {:#018x} RBX: {:#018x} RCX: {:018x} RDX: {:018x}",
            self.registers[Register::Rax as usize][index],
            self.registers[Register::Rbx as usize][index],
            self.registers[Register::Rcx as usize][index],
            self.registers[Register::Rdx as usize][index]
        );
        println!(
            "RIP: {:#018x}",
            self.registers[Register::Rip as usize][index],
        );

        self.hexdump(0xcff70, 0x20);
    }

    /// Step forward one step
    pub fn step_forward(&mut self) {
        timeloop::scoped_timer!(Timer::StepForward);

        let next_index = (self.index + 1).min(self.entries);
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

    pub fn exec_command(&mut self, command: &str) -> Result<(), ()> {
        match command.trim() {
            "next" | "n" => self.step_forward(),
            "sb" => self.step_backward(),
            "stats" => timeloop::print!(),
            "q" => return Err(()),
            x => println!("Unknown command: {x}"),
        }

        Ok(())
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

    pub fn goto_index(&mut self, new_index: usize) {
        timeloop::scoped_timer!(Timer::GotoIndex);

        self.update_memory_to(new_index);
        self.index = new_index;
    }

    /// Update the memory from the current index
    fn update_memory_to(&mut self, target_index: usize) {
        timeloop::scoped_timer!(Timer::UpdateMemory);

        if target_index >= self.index {
            timeloop::scoped_timer!(Timer::UpdateMemoryForward);
            // Update forward (easy path)
            for (i, diffs) in self.memory_diffs[self.index..target_index + 1]
                .iter()
                .enumerate()
            {
                for MemoryDataItem {
                    address,
                    bytes,
                    access,
                } in diffs.iter()
                {
                    self.memory
                        .set_bytes(self.index + i, address, bytes, access);
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
                for (i, diffs) in self.memory_diffs[target_index..self.index + 1]
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
                                writes.insert(address + offset, i + target_index);
                            }
                        }
                    }
                }
            }

            // Printing the current byte history
            for (addr, indexes) in self.memory.byte_history.iter() {
                // println!("{addr:#x} {indexes:?}");
            }

            for (addr, last_write) in writes.iter() {
                let Some((addresses, bytes)) = self.memory.byte_history.get_mut(addr) else {
                    panic!("Invalid byte history state");
                };

                let result = addresses.binary_search(last_write);

                /*
                println!(
                    "XXX {}..{} {addr:#x} {last_write} -> {result:?}",
                    self.index, target_index
                );
                */

                match result {
                    Err(0) => {
                        // If there was no prior byte in this address's history.
                        self.memory
                            .set_byte_state(*addr, memory::ByteState::Unknown);
                        self.memory.set_byte(*addr, 0);
                    }
                    Err(index) => {
                        // Get the last known byte before this address was written
                        let last_byte = bytes[index - 1];
                        // println!("  --> {last_write} ({last_byte:#x})");

                        self.memory.set_byte_state(*addr, memory::ByteState::Known);
                        self.memory.set_byte(*addr, last_byte);
                    }
                    Ok(mut index) => {
                        // println!("Found instr: {} Target: {}", addresses[index], target_index);
                        while index > 0 && addresses[index] > target_index {
                            // println!("  --> SUB INDEX");
                            index = index.saturating_sub(1);
                            /*
                            println!(
                                "  --> Found instr: {} Target: {} Byte: {:#x}",
                                addresses[index], target_index, bytes[index]
                            );
                            */
                        }

                        // Value was found, use this index as the last byte in the history
                        let last_byte = bytes[index];
                        // println!("  --> {last_write} ({last_byte:#x})");

                        self.memory.set_byte_state(*addr, memory::ByteState::Known);
                        self.memory.set_byte(*addr, last_byte);
                    }
                }
            }
        }
    }
}

pub enum CommandResult {
    Continue,
    Quit,
}

#[derive(Clone)]
pub struct TraceDiff<const REGISTER_ENTRIES: usize, const MEMORY_ENTRIES: usize> {
    pub register: [Register; REGISTER_ENTRIES],
    pub register_data: [u64; REGISTER_ENTRIES],
    pub memory_data: MemoryData<MEMORY_ENTRIES>,
}

impl<const REGISTER_ENTRIES: usize, const MEMORY_ENTRIES: usize>
    TraceDiff<REGISTER_ENTRIES, MEMORY_ENTRIES>
{
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
        bytes: [u8; 8],
        access: MemoryAccess,
    ) -> Result<(), Error> {
        timeloop::scoped_timer!(Timer::AddMemory);

        let new_memory = MemoryDiff { address, bytes };

        match &mut self.memory_data {
            MemoryData::Static((mem_values, size_of_values, mem_accesses)) => {
                for (index, mem_access) in mem_accesses.iter_mut().enumerate() {
                    // Look for an empty slot in the arrays for this entry
                    if !matches!(mem_access, MemoryAccess::None) {
                        continue;
                    }

                    // Found an empty slot, add this memory
                    *mem_access = access;
                    mem_values[index] = new_memory;
                    size_of_values[index] = num_bytes;
                    return Ok(());
                }
            }
            MemoryData::Dynamic((mem_addresses, mem_bytes, mem_accesses)) => {
                mem_addresses.push(address);
                mem_bytes.push(bytes[..num_bytes as usize].to_vec());
                mem_accesses.push(access);
                return Ok(());
            }
        }

        // Was not able to add to the static memory. Allocate and switch
        // to a dynamic memory
        let MemoryData::Static((mem_values, mem_sizes, mem_accesses)) = self.memory_data else {
            return Err(Error::AddMemory);
        };

        // Allocate and populate the memory values from the static allocations
        let mut new_data = Vec::new();
        let mut new_addresses = Vec::new();
        let mut new_accesses = mem_accesses.to_vec();
        for (index, data) in mem_values.iter().enumerate() {
            // Ensure that the access has actually been set
            if matches!(mem_accesses[index], MemoryAccess::None) {
                return Err(Error::AddMemory);
            };

            let MemoryDiff { address, bytes } = data;
            let data_size = mem_sizes[index] as usize;

            new_addresses.push(*address);
            new_data.push(bytes[..data_size].to_vec());
        }

        // Now add the requested memory
        new_addresses.push(address);
        new_data.push(bytes[..num_bytes as usize].to_vec());
        new_accesses.push(access);

        self.memory_data = MemoryData::Dynamic((new_addresses, new_data, new_accesses));

        Ok(())
    }
}

impl<const REGISTER_ENTRIES: usize, const MEMORY_ENTRIES: usize> std::default::Default
    for TraceDiff<REGISTER_ENTRIES, MEMORY_ENTRIES>
{
    fn default() -> Self {
        TraceDiff {
            register: [Register::None; REGISTER_ENTRIES],
            register_data: [0; REGISTER_ENTRIES],
            memory_data: MemoryData::default(),
        }
    }
}

impl<const REGISTER_ENTRIES: usize, const MEMORY_ENTRIES: usize> std::fmt::Debug
    for TraceDiff<REGISTER_ENTRIES, MEMORY_ENTRIES>
{
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

pub fn print_stats() {
    unsafe {
        TIMELOOP_PROFILER.print();
    }
}
