//! The bridge between the interpreter's memory interface and the [`Mmu`].
//!
//! [`VmMemory`] is a hybrid, and deliberately so. Guest process memory — the
//! default RAM space — goes through the MMU, where it gets mapping,
//! permissions, and faults. Every other space keeps the flat representation the
//! emulator already uses:
//!
//! * **Register** and **unique** spaces are not process memory at all. They are
//!   architectural state and per-instruction scratch, addressed by the lifter
//!   rather than the guest. There is no meaningful sense in which a write to
//!   `RAX` could be "unmapped", and putting a page lookup on that path would tax
//!   the single hottest operation in the interpreter.
//! * **Temporary** spaces are per-function IR storage, invisible to the guest.
//!
//! So the MMU is applied exactly where guest-visible addressing happens, and
//! nowhere else.

use qcode::{context::Context, space::MemorySpaceId};
use qcode_emulator::{
    DomainMemory, DomainValue, EmulatorErrorKind, EmulatorMemory, SizedValue,
};

use crate::{
    flat::FlatSpaces,
    mmu::{MemFault, Mmu},
};

/// Memory for a VM run: an [`Mmu`] for the RAM space, flat storage elsewhere.
#[derive(Default)]
pub struct VmMemory {
    /// Guest process memory.
    pub mmu: Mmu,
    /// Register, unique and temporary spaces, densely stored.
    flat: FlatSpaces,
    /// Which space the MMU backs. Resolved from the context on the first
    /// [`configure_spaces`](EmulatorMemory::configure_spaces); `None` until then,
    /// which routes everything to flat storage rather than guessing.
    ram: Option<MemorySpaceId>,
    /// The last fault produced by an MMU access.
    ///
    /// [`EmulatorErrorKind`] can only carry "a read failed at this address", so
    /// the precise cause would be lost on the way out of the interpreter. The VM
    /// layer takes the fault from here to build an exact exit, which is what
    /// makes a fault a resumable value rather than an abort. Cleared by
    /// [`take_fault`](Self::take_fault).
    fault: Option<MemFault>,
}

impl VmMemory {
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns and clears the fault recorded by the most recent failed access.
    pub fn take_fault(&mut self) -> Option<MemFault> {
        self.fault.take()
    }

    /// Whether `space` is the guest RAM backed by the MMU.
    fn is_ram(&self, space: MemorySpaceId) -> bool {
        self.ram == Some(space)
    }

    /// Records `fault` and converts it into the error the interpreter
    /// understands. The address is preserved in both, so a consumer that never
    /// looks at [`take_fault`](Self::take_fault) still gets a truthful error.
    fn record(&mut self, fault: MemFault, writing: bool) -> EmulatorErrorKind {
        self.fault = Some(fault);
        if writing {
            EmulatorErrorKind::MemoryWriteError(fault.addr)
        } else {
            EmulatorErrorKind::MemoryReadError(fault.addr)
        }
    }
}

impl DomainMemory for VmMemory {
    type V = SizedValue;

    fn read(
        &self,
        space: MemorySpaceId,
        addr: Self::V,
        size: usize,
    ) -> Result<Self::V, EmulatorErrorKind> {
        if !self.is_ram(space) {
            let bits = self.flat.read_u128(space, addr.value()?, size)?;
            return Ok(SizedValue::from_bits(bits, size));
        }
        let addr = addr.value()?;
        // A value wider than 16 bytes cannot be held by `SizedValue`; the flat
        // backend truncates the same way, so the two agree.
        let mut bytes = vec![0u8; size.min(16)];
        // `&self` cannot record the fault; the error still carries the address,
        // and every faulting path the VM actually resumes from goes through a
        // `&mut self` write or through `read_mut` below.
        self.mmu
            .read(addr, &mut bytes)
            .map_err(|fault| EmulatorErrorKind::MemoryReadError(fault.addr))?;
        let mut bits = 0u128;
        for (index, byte) in bytes.iter().enumerate() {
            bits |= u128::from(*byte) << (index * 8);
        }
        Ok(SizedValue::from_bits(bits, size))
    }

    fn write(
        &mut self,
        space: MemorySpaceId,
        addr: Self::V,
        size: usize,
        value: Self::V,
    ) -> Result<(), EmulatorErrorKind> {
        if !self.is_ram(space) {
            return self
                .flat
                .entry(space)
                .write_u128(addr.value()?, size, value.as_bits());
        }
        let addr = addr.value()?;
        let bits = value.as_bits();
        let width = size.min(16);
        let bytes: Vec<u8> = (0..width).map(|i| (bits >> (i * 8)) as u8).collect();
        self.mmu
            .write(addr, &bytes)
            .map_err(|fault| self.record(fault, true))
    }
}

impl EmulatorMemory for VmMemory {
    fn configure_spaces(&mut self, ctx: &Context<'_>) {
        // The guest's RAM is the specification's default space: the one a bare
        // `Load`/`Store` addresses, and the only one a guest pointer refers to.
        self.ram = Some(MemorySpaceId::Shared(ctx.shared.default_space));
        self.flat.configure(ctx);
    }

    fn read_bytes(
        &self,
        space: MemorySpaceId,
        addr: u64,
        size: usize,
    ) -> Result<Vec<u8>, EmulatorErrorKind> {
        if !self.is_ram(space) {
            return self.flat.read_bytes(space, addr, size);
        }
        let mut bytes = vec![0u8; size];
        self.mmu
            .read(addr, &mut bytes)
            .map_err(|fault| EmulatorErrorKind::MemoryReadError(fault.addr))?;
        Ok(bytes)
    }

    fn write_bytes(
        &mut self,
        space: MemorySpaceId,
        addr: u64,
        bytes: &[u8],
    ) -> Result<(), EmulatorErrorKind> {
        if !self.is_ram(space) {
            return self.flat.entry(space).write_bytes(addr, bytes);
        }
        self.mmu
            .write(addr, bytes)
            .map_err(|fault| self.record(fault, true))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mmu::{FaultKind, PAGE_SIZE, perm};
    use qcode::space::SpaceId;

    /// A context has a RAM default space; the register space is a second one.
    fn context() -> Context<'static> {
        Context::default()
    }

    fn configured() -> (Context<'static>, VmMemory) {
        let ctx = context();
        let mut memory = VmMemory::new();
        memory.configure_spaces(&ctx);
        (ctx, memory)
    }

    fn ram(ctx: &Context<'_>) -> MemorySpaceId {
        MemorySpaceId::Shared(ctx.shared.default_space)
    }

    #[test]
    fn ram_accesses_go_through_the_mmu() {
        let (ctx, mut memory) = configured();
        memory.mmu.map(0x1000, PAGE_SIZE, perm::RW_INIT).unwrap();

        let addr = SizedValue::from_u64(0x1000);
        memory
            .write(ram(&ctx), addr, 4, SizedValue::from_bits(0xdead_beef, 4))
            .unwrap();
        let read = memory.read(ram(&ctx), addr, 4).unwrap();
        assert_eq!(read.as_bits(), 0xdead_beef);
        // The bytes really are in the MMU, little-endian.
        let mut raw = [0; 4];
        memory.mmu.read(0x1000, &mut raw).unwrap();
        assert_eq!(raw, [0xef, 0xbe, 0xad, 0xde]);
    }

    #[test]
    fn unmapped_ram_write_faults_and_is_recorded() {
        let (ctx, mut memory) = configured();
        let addr = SizedValue::from_u64(0x1000);
        let error = memory
            .write(ram(&ctx), addr, 4, SizedValue::from_bits(1, 4))
            .unwrap_err();
        assert!(matches!(error, EmulatorErrorKind::MemoryWriteError(0x1000)));
        assert_eq!(
            memory.take_fault(),
            Some(MemFault {
                kind: FaultKind::WriteUnmapped,
                addr: 0x1000
            })
        );
        // Taking the fault clears it.
        assert_eq!(memory.take_fault(), None);
    }

    #[test]
    fn read_only_ram_refuses_a_write() {
        let (ctx, mut memory) = configured();
        memory.mmu.map(0x1000, PAGE_SIZE, perm::RX_INIT).unwrap();
        let addr = SizedValue::from_u64(0x1000);
        memory
            .write(ram(&ctx), addr, 1, SizedValue::from_bits(1, 1))
            .unwrap_err();
        assert_eq!(memory.take_fault().map(|f| f.kind), Some(FaultKind::WritePerm));
    }

    #[test]
    fn non_ram_spaces_bypass_the_mmu_entirely() {
        let ctx = context();
        let mut memory = VmMemory::new();
        memory.configure_spaces(&ctx);

        // A space that is not the default one is flat: it needs no mapping and
        // takes no faults.
        let register = MemorySpaceId::Shared(SpaceId::from(1));
        let addr = SizedValue::from_u64(0x40);
        memory
            .write(register, addr, 8, SizedValue::from_bits(0x1234, 8))
            .unwrap();
        assert_eq!(memory.read(register, addr, 8).unwrap().as_bits(), 0x1234);
        assert_eq!(memory.mmu.resident_pages(), 0);
        assert_eq!(memory.take_fault(), None);
    }

    #[test]
    fn byte_level_access_routes_the_same_way() {
        let (ctx, mut memory) = configured();
        memory.mmu.map(0x2000, PAGE_SIZE, perm::RW_INIT).unwrap();
        memory
            .write_bytes(ram(&ctx), 0x2000, &[1, 2, 3, 4])
            .unwrap();
        assert_eq!(
            memory.read_bytes(ram(&ctx), 0x2000, 4).unwrap(),
            vec![1, 2, 3, 4]
        );
    }

    #[test]
    fn before_configuration_nothing_is_routed_to_the_mmu() {
        // Guessing a RAM space before the context is known would silently send
        // register traffic through the MMU and fault on it.
        let mut memory = VmMemory::new();
        let addr = SizedValue::from_u64(0x1000);
        let space = MemorySpaceId::Shared(SpaceId::from(0));
        memory
            .write(space, addr, 4, SizedValue::from_bits(7, 4))
            .unwrap();
        assert_eq!(memory.mmu.resident_pages(), 0);
    }
}
