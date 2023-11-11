use crate::{InstrIndex, MemoryAccess};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// The size of each memory page that the debugger knows about. Must
/// be a power of two.
const PAGE_SIZE: u64 = 8 * 1024;

/// The mask used to get the beginning of a page using a bitwise and.
///
/// Example:
/// Address: 0x1234_a234
/// Mask:    0xffff_f000
/// Page:    0x1234_a000
const OFFSET_MASK: u64 = PAGE_SIZE - 1;
const PAGE_MASK: u64 = !OFFSET_MASK;

// Ensure page size is power of two
const _: () = assert!(PAGE_SIZE.is_power_of_two());

#[derive(Serialize, Deserialize, Debug, Default, Copy, Clone, PartialEq, Eq)]
pub enum ByteState {
    #[default]
    Unknown,
    Known,
}

/// A memory backing
#[derive(Serialize, Deserialize, Default, Clone, PartialEq, Eq)]
pub struct Memory {
    /// Lookup table from masked address to the index into the memory list.
    ///
    /// Example:
    /// 0x1234_0000 -> 0 => self.memory[0][8] = 0x1234_0008
    /// 0xdead_0000 -> 1 => self.memory[1][8] = 0xdead_0008
    pub lookup_table_addresses: Vec<u64>,

    /// The state of each byte in memory
    pub byte_state: Vec<Vec<ByteState>>,

    /// The memory tables in the debugger
    pub memory: Vec<Vec<u8>>,
}

impl Memory {
    /// Reset all of the memory bytes, but keep the lookup table structure allocated
    pub fn soft_reset(&mut self) {
        timeloop::scoped_timer!(Timer::Memory_SoftReset);

        self.byte_state
            .iter_mut()
            .for_each(|page| *page = vec![ByteState::Unknown; PAGE_SIZE as usize])
    }

    /// Set the byte at the given address to the given state
    pub fn set_byte_state(&mut self, address: u64, state: ByteState) {
        let page_index = self.get_page_index(address);
        let offset = (address & OFFSET_MASK) as usize;
        self.byte_state[page_index][offset] = state;
    }

    /// Set the current memory address with the given bytes
    pub fn set_bytes(
        &mut self,
        instr_index: InstrIndex,
        address: u64,
        bytes: &[u8],
        access: MemoryAccess,
        memory_reads: &mut BTreeMap<u64, Vec<InstrIndex>>,
        memory_writes: &mut BTreeMap<u64, Vec<InstrIndex>>,
        byte_history: &mut BTreeMap<u64, (Vec<InstrIndex>, Vec<u8>)>,
    ) {
        if self.is_straddling_page(address, bytes.len() as u64) {
            panic!("Address straddles memory: {address:#x} {bytes:x?}");
        }

        // Get the page index for this address
        let page_index = self.get_page_index(address);

        // Get the offset
        let offset = (address & OFFSET_MASK) as usize;

        // Copy the known bytes into memory and set them as known
        self.memory[page_index][offset..offset + bytes.len()].copy_from_slice(bytes);
        self.byte_state[page_index][offset..offset + bytes.len()]
            .copy_from_slice(&vec![ByteState::Known; bytes.len()]);

        // Initialize the history for each address
        for offset in 0..bytes.len() {
            let curr_addr = address + offset as u64;
            byte_history.entry(curr_addr).or_default();

            if matches!(access, MemoryAccess::Read) {
                memory_reads.entry(curr_addr).or_default();
            } else if matches!(access, MemoryAccess::Read) {
                memory_writes.entry(curr_addr).or_default();
            }
        }

        // Set the requested bytes to the memory
        for (index, byte) in bytes.iter().enumerate() {
            // Setup the byte_history
            //
            // If this address hasn't been seen yet, and it was just read, then that byte starts.
            let curr_addr = address + index as u64;

            // Mark this byte in the history for this instruction index
            let (indexes, bytes) = byte_history.get_mut(&curr_addr).unwrap();

            if *indexes.last().unwrap_or(&0) <= instr_index {
                indexes.push(instr_index);
                bytes.push(*byte);

                if matches!(access, MemoryAccess::Read) {
                    memory_reads.entry(curr_addr).or_default().push(instr_index);
                } else if matches!(access, MemoryAccess::Write) {
                    memory_writes
                        .entry(curr_addr)
                        .or_default()
                        .push(instr_index);
                }
            }
        }
    }

    /// Set the given byte to the given address
    pub fn set_byte(&mut self, address: u64, byte: u8) {
        timeloop::scoped_timer!(Timer::Memory_SetByte);

        // Get the page index for this address
        let page_index = self.get_page_index(address);

        // Set the single byte specifically
        let offset = (address & OFFSET_MASK) as usize;
        self.memory[page_index][offset] = byte;
        self.byte_state[page_index][offset] = ByteState::Known;
    }

    /// Returns true if address..address + n straddles the given PAGE_SIZE
    pub fn is_straddling_page(&self, address: u64, n: u64) -> bool {
        timeloop::scoped_timer!(Timer::Memory_IsStraddlingPage);

        let curr_page = address & PAGE_MASK;
        let final_page = (address + n - 1) & PAGE_MASK;
        curr_page != final_page
    }

    /// Lookup the page index for the given address. If not found, allocates a page and returns
    /// the new page's address
    pub fn get_page_index(&mut self, address: u64) -> usize {
        timeloop::scoped_timer!(Timer::Memory_GetPageIndex);

        // Lookup the page index for this address
        for (index, page_table) in self.lookup_table_addresses.iter().enumerate() {
            if *page_table == address & PAGE_MASK {
                return index;
            }
        }

        // Did not find a page table that we know about, allocate one
        let index = self.allocate_page();
        self.lookup_table_addresses.push(address & PAGE_MASK);
        index
    }

    pub fn read(&mut self, address: u64, n: usize) -> Vec<Option<u8>> {
        timeloop::scoped_timer!(Timer::Memory_Read);

        if self.is_straddling_page(address, n as u64) {
            panic!("Read straddles memory: {address:#x} {n}");
        }

        let page_index = self.get_page_index(address);
        let offset = (address & OFFSET_MASK) as usize;

        let mut result = Vec::with_capacity(n);
        for byte_index in offset..offset + n {
            log::debug!(
                "Reading {page_index} {byte_index:#x} -> {:?}",
                self.byte_state[page_index][byte_index]
            );

            if matches!(self.byte_state[page_index][byte_index], ByteState::Unknown) {
                result.push(None);
            } else {
                result.push(Some(self.memory[page_index][byte_index]));
            }
        }

        result
    }

    /// Allocate a page of memory and return the index into the lookup table for this memory
    pub fn allocate_page(&mut self) -> usize {
        timeloop::scoped_timer!(Timer::Memory_AllocatePage);
        let new_memory = vec![0; PAGE_SIZE as usize];
        let new_memory_states = vec![ByteState::Unknown; PAGE_SIZE as usize];

        let new_index = self.memory.len();
        self.memory.push(new_memory);
        self.byte_state.push(new_memory_states);
        new_index
    }

    pub fn hexdump(&mut self, address: u64, n: usize) {
        timeloop::scoped_timer!(Timer::Memory_Hexdump);

        let bytes = self.read(address, n);

        print!("{address:#018x} | ");
        for (index, byte) in bytes.iter().enumerate() {
            if index > 0 && index % 0x10 == 0 {
                println!();
                print!("{:#018x} | ", address as usize + index);
            }

            match byte {
                Some(byte) => print!("{byte:02x} "),
                None => print!("?? "),
            }
        }

        println!();
    }

    pub fn diff(&self, other: &Memory) {
        for (page_index, page) in self.byte_state.iter().enumerate() {
            for (byte_index, access) in page.iter().enumerate() {
                if matches!(access, ByteState::Unknown) {
                    continue;
                }

                log::debug!(
                    "Curr state {:?} Other state: {:?}",
                    self.byte_state[page_index][byte_index],
                    other.byte_state[page_index][byte_index]
                );

                log::debug!("Page: {page_index:#x} Byte index: {byte_index:#x}");
                log::debug!(
                    "Curr byte: {:x?} Other byte: {:x?}",
                    self.memory[page_index][byte_index],
                    other.memory[page_index][byte_index]
                );

                assert!(
                    self.memory[page_index][byte_index] == other.memory[page_index][byte_index]
                );
            }
        }
    }
}
