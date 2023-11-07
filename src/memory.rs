use crate::Timer;

/// The size of each memory page that the debugger knows about. Must
/// be a power of two.
const PAGE_SIZE: u64 = 4096;

/// The mask used to get the beginning of a page using a bitwise and.
///
/// Example:
/// Address: 0x1234_a234
/// Mask:    0xffff_f000
/// Page:    0x1234_a000
const OFFSET_MASK: u64 = PAGE_SIZE - 1;
const PAGE_MASK: u64 = !OFFSET_MASK;

/// Constant check if the given number is a power of two
const fn is_power_of_two(n: u64) -> bool {
    n != 0 && (n & (n - 1)) == 0
}

// Ensure page size is power of two
const _: () = assert!(is_power_of_two(PAGE_SIZE));

/// A memory backing
#[derive(Default)]
pub struct Memory {
    /// Lookup table from masked address to the index into the memory list.
    ///
    /// Example:
    /// 0x1234_0000 -> 0 => self.memory[0][8] = 0x1234_0008
    /// 0xdead_0000 -> 1 => self.memory[1][8] = 0xdead_0008
    pub lookup_table_addresses: Vec<u64>,

    /// The memory tables in the debugger
    pub memory: Vec<Vec<Option<u8>>>,
}

impl Memory {
    /// Reset all of the memory bytes, but keep the lookup table structure allocated
    pub fn soft_reset(&mut self) {
        self.memory
            .iter_mut()
            .for_each(|page| *page = vec![None; PAGE_SIZE as usize])
    }

    /// Set the current memory address with the given bytes
    pub fn set_bytes(&mut self, address: u64, bytes: &[u8]) {
        timeloop::scoped_timer!(Timer::Memory_SetBytes);

        if self.is_straddling_page(address, bytes.len() as u64) {
            panic!("Address straddles memory: {address:#x} {bytes:x?}");
        }

        // Get the page index for this address
        let page_index = self.get_page_index(address);

        // Get the current page of memory containing this address
        let curr_mem_page = &mut self.memory[page_index];

        // Set the requested bytes to the memory
        let offset = (address & OFFSET_MASK) as usize;
        for (index, byte) in bytes.iter().enumerate() {
            curr_mem_page[offset + index] = Some(*byte);
        }
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

    pub fn read(&mut self, address: u64, n: usize) -> &[Option<u8>] {
        timeloop::scoped_timer!(Timer::Memory_Read);

        if self.is_straddling_page(address, n as u64) {
            panic!("Read straddles memory: {address:#x} {n}");
        }

        let page_index = self.get_page_index(address);
        let offset = (address & OFFSET_MASK) as usize;
        &self.memory[page_index][offset..offset + n]
    }

    /// Allocate a page of memory and return the index into the lookup table for this memory
    pub fn allocate_page(&mut self) -> usize {
        timeloop::scoped_timer!(Timer::Memory_AllocatePage);
        let new_memory = vec![None; PAGE_SIZE as usize];
        let new_index = self.memory.len();
        self.memory.push(new_memory);
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
}
