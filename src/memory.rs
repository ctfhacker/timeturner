use crate::{InstrIndex, MemoryAccess};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::mem::size_of;

/// The size of each memory page that the debugger knows about. Must
/// be a power of two.
const PAGE_SIZE: u64 = 64;

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

// 0 - Unknown byte state, 1 - Known byte state
type PackedByteState = u128;

// Get the size of the packed byte state
const PACKED_BYTE_STATE_SIZE: usize = size_of::<PackedByteState>();

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
    // pub byte_state: Vec<Vec<PackedByteState>>,

    /// The memory tables in the debugger
    pub memory: Vec<Vec<u8>>,
}

impl Memory {
    /// Reset all of the memory bytes, but keep the lookup table structure allocated
    pub fn soft_reset(&mut self) {
        timeloop::scoped_timer!(crate::Timer::Memory_SoftReset);

        /*
        self.byte_state
            .iter_mut()
            .for_each(|page| *page = vec![0; PAGE_SIZE as usize / PACKED_BYTE_STATE_SIZE])
        */
    }

    /// Set the byte at the given address to the given state
    /*
    pub fn set_byte_state(&mut self, address: u64, state: ByteState) {
        let page_index = self.get_page_index(address);

        let offset = (address & OFFSET_MASK) as usize;
        let byte_index = offset / PACKED_BYTE_STATE_SIZE;
        let bit_index = offset % PACKED_BYTE_STATE_SIZE;

        if matches!(state, ByteState::Known) {
            self.byte_state[page_index][byte_index] |= 1 << bit_index;
        } else {
            self.byte_state[page_index][byte_index] &= !(1 << bit_index);
        }
    }
    */

    /// Set the byte at the given address to the given state
    /*
    pub fn get_byte_state(&mut self, address: u64) -> ByteState {
        let page_index = self.get_page_index(address);

        let offset = (address & OFFSET_MASK) as usize;
        let byte_index = offset / PACKED_BYTE_STATE_SIZE;
        let bit_index = offset % PACKED_BYTE_STATE_SIZE;

        if self.byte_state[page_index][byte_index] & (1 << bit_index) > 0 {
            ByteState::Known
        } else {
            ByteState::Unknown
        }
    }
    */

    pub fn write_bytes(&mut self, mut address: u64, bytes: &[u8]) {
        timeloop::scoped_timer!(crate::Timer::Memory_WriteBytes);

        // log::info!("Writing: {address:#x} bytes {bytes:x?}");
        let mut byte_pos = 0;
        let n = bytes.len();
        let ending_addr = address + n as u64;

        while address < ending_addr {
            let next_page = ((address + PAGE_SIZE) & PAGE_MASK).min(ending_addr);
            let curr_len = next_page - address;

            let page_index = self.get_page_index(address);
            let offset = (address & OFFSET_MASK) as usize;

            // Copy the bytes into this page
            /*
            log::info!(
                "WRITING {address:#x} {:x?}",
                &bytes[byte_pos..byte_pos + curr_len as usize]
            );
            */

            self.memory[page_index][offset..offset + curr_len as usize]
                .copy_from_slice(&bytes[byte_pos..byte_pos + curr_len as usize]);

            // Set the state of these bytes to known
            /*
            for addr in address..address + curr_len {
                self.set_byte_state(addr, ByteState::Known);
            }
            */

            address += curr_len;
            byte_pos += curr_len as usize;
        }
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
        ready_vec_bytes: &mut Vec<Vec<u8>>,
        ready_vec_instr_index: &mut Vec<Vec<InstrIndex>>,
    ) {
        // Copy the known bytes into memory and set them as known
        timeloop::scoped_timer!(crate::Timer::Memory_SetBytes1);
        // self.write_bytes(address, bytes);

        // Set the requested bytes to the memory
        for (index, byte) in bytes.iter().enumerate() {
            // Setup the byte_history
            //
            // If this address hasn't been seen yet, and it was just read, then that byte starts.
            let curr_addr = address + index as u64;

            // Mark this byte in the history for this instruction index
            let (indexes, bytes) = byte_history.entry(curr_addr).or_insert_with(|| {
                (
                    ready_vec_instr_index.pop().unwrap_or_default(),
                    ready_vec_bytes.pop().unwrap_or_default(),
                )
            });

            if *indexes.last().unwrap_or(&0) <= instr_index {
                indexes.push(instr_index);
                bytes.push(*byte);

                if matches!(access, MemoryAccess::Read) {
                    memory_reads
                        .entry(curr_addr)
                        .or_insert_with(|| ready_vec_instr_index.pop().unwrap_or_default())
                        .push(instr_index);
                } else if matches!(access, MemoryAccess::Write) {
                    memory_writes
                        .entry(curr_addr)
                        .or_insert_with(|| ready_vec_instr_index.pop().unwrap_or_default())
                        .push(instr_index);
                }
            }
        }
    }

    /// Set the given byte to the given address
    pub fn set_byte(&mut self, address: u64, byte: u8) {
        timeloop::scoped_timer!(crate::Timer::Memory_SetByte);

        // Get the page index for this address
        let page_index = self.get_page_index(address);
        let offset = (address & OFFSET_MASK) as usize;
        self.memory[page_index][offset] = byte;

        // Set the byte state as known for this byte
        // self.set_byte_state(address, ByteState::Known);
    }

    /// Lookup the page index for the given address. If not found, allocates a page and returns
    /// the new page's address
    pub fn get_page_index(&mut self, address: u64) -> usize {
        let page_base = address & PAGE_MASK;

        // Lookup the page index for this address
        for (index, page_table) in self.lookup_table_addresses.iter().enumerate() {
            if *page_table == page_base {
                return index;
            }
        }

        // Did not find a page table that we know about, allocate one
        let index = self.allocate_page();
        self.lookup_table_addresses.push(page_base);
        index
    }

    /// Read n bytes from the given address
    pub fn read(&mut self, mut address: u64, n: usize) -> Vec<u8> {
        timeloop::scoped_timer!(crate::Timer::Memory_Read);

        let mut result = Vec::with_capacity(n);
        let ending_addr = address + n as u64;

        while address < ending_addr {
            let next_page = ((address + PAGE_SIZE) & PAGE_MASK).min(ending_addr);
            let curr_len = next_page - address;
            let page_index = self.get_page_index(address);

            for i in 0..curr_len {
                let curr_addr = address + i as u64;
                let byte_index = (curr_addr & OFFSET_MASK) as usize;

                /*
                let access = self.get_byte_state(curr_addr);

                if matches!(access, ByteState::Unknown) {
                    result.push(None);
                } else {
                    result.push(Some(self.memory[page_index][byte_index]));
                }
                */
                result.push(self.memory[page_index][byte_index]);
            }

            // Update the address to the next page
            address += curr_len;
        }

        result
    }

    /// Allocate a page of memory and return the index into the lookup table for this memory
    pub fn allocate_page(&mut self) -> usize {
        timeloop::scoped_timer!(crate::Timer::Memory_AllocatePage);
        let new_memory = vec![0; PAGE_SIZE as usize];
        let new_memory_states = vec![0; PAGE_SIZE as usize / PACKED_BYTE_STATE_SIZE];

        let new_index = self.memory.len();
        self.memory.push(new_memory);
        // self.byte_state.push(new_memory_states);
        new_index
    }

    pub fn hexdump(&mut self, address: u64, n: usize) {
        timeloop::scoped_timer!(crate::Timer::Memory_Hexdump);

        let bytes = self.read(address, n);

        print!("{address:#018x} | ");
        for (index, byte) in bytes.iter().enumerate() {
            if index > 0 && index % 0x10 == 0 {
                println!();
                print!("{:#018x} | ", address as usize + index);
            }

            /*
            match byte {
                Some(byte) => print!("{byte:02x} "),
                None => print!("?? "),
            }
            */
            print!("{byte:02x} ");
        }

        println!();
    }

    pub fn diff(&self, other: &Memory) {
        for (page_index, page) in self.memory.iter().enumerate() {
            for (byte_index, byte) in page.iter().enumerate() {
                assert!(
                    self.memory[page_index][byte_index] == other.memory[page_index][byte_index]
                );
            }
        }
    }
}
