//! What context an instruction is decoded with.
//!
//! SLEIGH decoding is pure: the bytes and a context register in, an
//! instruction out. What differs between consumers is where that context
//! comes from, and lifting keeps that choice separate from where the IR
//! goes:
//!
//! - A [`FixedDecoder`] decodes every address with one configured context.
//!   An instruction's `globalset` effects are reported on the instruction and
//!   never applied, so decoding an address is independent of what was
//!   decoded before it. This is the default: an emulator fetching on demand,
//!   a corpus replay and an analysis of isolated offsets all want it.
//! - A [`LinearDecoder`] sweeps forward through a function, committing each
//!   instruction's context effects for the addresses after it. It refuses an
//!   address other than the one it expects next, so a sweep's history is
//!   never applied to an unrelated offset by accident.
//! - A caller that models context itself decodes with [`Decoder`] and hands
//!   the lifter the decoded instruction.
//!
//! The [`ContextDatabase`] under the linear decoder is address-keyed static
//! analysis state. It is not a model of an emulated machine's mode changes,
//! which a VM keys explicitly.

use sleigh::{
    CompiledSpec, ContextBytes, ContextDatabase, ContextError, DecodeError, Decoder, Instruction,
    SpecFingerprint,
};

/// Decodes any address with one fixed context. See the [module
/// documentation](self).
pub struct FixedDecoder<'spec> {
    decoder: Decoder<'spec>,
    context: ContextBytes,
}

impl<'spec> FixedDecoder<'spec> {
    /// Decodes with `spec`'s default context.
    pub fn new(spec: &'spec CompiledSpec) -> Self {
        Self {
            decoder: Decoder::new(spec),
            context: spec.new_context(),
        }
    }

    /// Decodes with `context`, which must be `spec`'s context length.
    pub fn with_context(
        spec: &'spec CompiledSpec,
        context: ContextBytes,
    ) -> Result<Self, ContextError> {
        let expected = spec.new_context().len();
        if context.len() != expected {
            return Err(ContextError::InvalidLength {
                expected,
                actual: context.len(),
            });
        }
        Ok(Self {
            decoder: Decoder::new(spec),
            context,
        })
    }

    /// The context every decode uses.
    pub fn context(&self) -> &ContextBytes {
        &self.context
    }

    /// Decodes the instruction at `address` from `bytes`, which start there.
    pub fn decode<'bytes>(
        &self,
        address: u64,
        bytes: &'bytes [u8],
    ) -> Result<Instruction<'spec, 'bytes>, DecodeError> {
        self.decoder.decode_one(address, bytes, &self.context)
    }
}

/// Why a linear decoder refused an address.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinearDecodeError {
    /// The address is not the one the sweep reaches next.
    OutOfOrder {
        expected: u64,
        actual: u64,
    },
    /// The instruction was decoded by another specification, so its context
    /// effects mean nothing to this sweep.
    ForeignSpecification {
        expected: SpecFingerprint,
        actual: SpecFingerprint,
    },
    Decode(DecodeError),
}

impl std::fmt::Display for LinearDecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OutOfOrder { expected, actual } => {
                write!(
                    f,
                    "expected {expected:#x} next in the sweep, got {actual:#x}"
                )
            }
            Self::ForeignSpecification { expected, actual } => {
                write!(
                    f,
                    "the instruction was decoded by specification {actual}, the sweep uses {expected}"
                )
            }
            Self::Decode(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for LinearDecodeError {}

impl From<DecodeError> for LinearDecodeError {
    fn from(error: DecodeError) -> Self {
        Self::Decode(error)
    }
}

/// Decodes a forward sweep, carrying each instruction's context effects to
/// the addresses after it. See the [module documentation](self).
pub struct LinearDecoder<'spec> {
    decoder: Decoder<'spec>,
    fingerprint: SpecFingerprint,
    contexts: ContextDatabase,
    next: u64,
}

impl<'spec> LinearDecoder<'spec> {
    /// A sweep starting at `start` from `spec`'s default context.
    pub fn new(spec: &'spec CompiledSpec, start: u64) -> Self {
        Self {
            decoder: Decoder::new(spec),
            fingerprint: spec.fingerprint(),
            contexts: ContextDatabase::new(spec),
            next: start,
        }
    }

    /// Replaces the context the sweep starts every address from.
    pub fn set_default_context(&mut self, context: ContextBytes) -> Result<(), ContextError> {
        self.contexts.set_default_context(context)
    }

    /// The address the sweep decodes next.
    pub fn next_address(&self) -> u64 {
        self.next
    }

    /// Decodes the next instruction of the sweep, which must be at
    /// `address`, with the context the sweep so far established there.
    ///
    /// Nothing is committed: [`advance`](Self::advance) does that once the
    /// caller has lowered the instruction, so a failed lift leaves the sweep
    /// where it was. Decoding takes `&self`, so the same address may be
    /// decoded more than once — with a different context, say — and the
    /// sweep advances past whichever instruction the caller commits.
    pub fn decode<'bytes>(
        &self,
        address: u64,
        bytes: &'bytes [u8],
    ) -> Result<Instruction<'spec, 'bytes>, LinearDecodeError> {
        if address != self.next {
            return Err(LinearDecodeError::OutOfOrder {
                expected: self.next,
                actual: address,
            });
        }
        Ok(self
            .decoder
            .decode_one(address, bytes, &self.contexts.context_at(address))?)
    }

    /// Commits `instruction`'s context effects and moves the sweep past it.
    /// The instruction must be the sweep's next one, decoded by its
    /// specification: an instruction of another specification is refused
    /// before anything is committed.
    pub fn advance(&mut self, instruction: &Instruction<'_, '_>) -> Result<(), LinearDecodeError> {
        let expected = self.fingerprint;
        let actual = instruction.spec().fingerprint();
        if actual != expected {
            return Err(LinearDecodeError::ForeignSpecification { expected, actual });
        }
        if instruction.address() != self.next {
            return Err(LinearDecodeError::OutOfOrder {
                expected: self.next,
                actual: instruction.address(),
            });
        }
        self.contexts.apply(instruction);
        self.next = instruction.next_address();
        Ok(())
    }

    /// Forgets every committed effect and starts the sweep over at `start`,
    /// from the configured default context.
    pub fn restart(&mut self, start: u64) {
        self.contexts.clear();
        self.next = start;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sleigh::{Compiler, SourceDb};

    /// A specification whose `mode` context bit selects between two
    /// instructions for the same byte, and whose `switch` instruction sets it
    /// for what follows.
    fn spec() -> CompiledSpec {
        spec_with("")
    }

    /// The same specification with `extra` appended: a look-alike that
    /// decodes the same bytes to the same text, under another fingerprint.
    fn spec_with(extra: &str) -> CompiledSpec {
        let mut sources = SourceDb::new();
        let root = sources.add_file(
            "modes.slaspec",
            format!(
                "define endian=little;
             define space ram type=ram_space size=4 default;
             define space register type=register_space size=4;
             define register offset=0 size=4 [ r0 contextreg ];
             define context contextreg mode=(0,0);
             define token instr(8) op=(0,7);
             :one is mode=0 & op=1 {{ r0 = 1:4; }}
             :two is mode=1 & op=1 {{ r0 = 2:4; }}
             :switch is op=2 [ mode=1; globalset(inst_next, mode); ] {{ r0 = 0:4; }}
             {extra}"
            ),
        );
        Compiler::new(&mut sources).compile(root).unwrap()
    }

    #[test]
    fn a_linear_decoder_refuses_an_instruction_of_another_specification() {
        let spec = spec();
        let look_alike = spec_with(":three is op=3 { r0 = 3:4; }");
        let mut decoder = LinearDecoder::new(&spec, 0x1000);
        // Decoded at the expected address, by a decoder of the other
        // specification: the text agrees, the provenance does not.
        let foreign = Decoder::new(&look_alike)
            .decode_one(0x1000, &[2, 1], &look_alike.new_context())
            .unwrap();
        assert_eq!(foreign.to_string(), "switch");
        assert_eq!(
            decoder.advance(&foreign).unwrap_err(),
            LinearDecodeError::ForeignSpecification {
                expected: spec.fingerprint(),
                actual: look_alike.fingerprint(),
            }
        );
        // Nothing was committed: the sweep is where it was, in its default
        // context, so the next byte still decodes as `one`.
        assert_eq!(decoder.next_address(), 0x1000);
        let own = decoder.decode(0x1000, &[2, 1]).unwrap();
        decoder.advance(&own).unwrap();
        assert_eq!(decoder.decode(0x1001, &[1]).unwrap().to_string(), "two");
    }

    #[test]
    fn a_fixed_decoder_never_applies_effects() {
        let spec = spec();
        let decoder = FixedDecoder::new(&spec);
        let switch = decoder.decode(0x1000, &[2, 1]).unwrap();
        assert_eq!(switch.to_string(), "switch");
        assert_eq!(decoder.decode(0x1001, &[1]).unwrap().to_string(), "one");
    }

    #[test]
    fn a_linear_decoder_commits_on_advance_and_rejects_out_of_order() {
        let spec = spec();
        let mut decoder = LinearDecoder::new(&spec, 0x1000);
        let switch = decoder.decode(0x1000, &[2, 1]).unwrap();
        assert_eq!(
            decoder.decode(0x1001, &[1]).unwrap_err(),
            LinearDecodeError::OutOfOrder {
                expected: 0x1000,
                actual: 0x1001
            }
        );
        // Nothing is committed before the caller says so.
        decoder.advance(&switch).unwrap();
        assert_eq!(decoder.next_address(), 0x1001);
        let two = decoder.decode(0x1001, &[1]).unwrap();
        assert_eq!(two.to_string(), "two");
        decoder.advance(&two).unwrap();

        decoder.restart(0x1000);
        assert_eq!(decoder.decode(0x1000, &[1]).unwrap().to_string(), "one");
        assert_eq!(
            decoder.advance(&two).unwrap_err(),
            LinearDecodeError::OutOfOrder {
                expected: 0x1000,
                actual: 0x1001
            }
        );
    }

    #[test]
    fn a_configured_context_is_the_fixed_baseline() {
        let spec = spec();
        let mut context = spec.new_context();
        let mode = spec.field("mode").unwrap().id;
        spec.set_context_field(&mut context, mode, 1).unwrap();
        let decoder = FixedDecoder::with_context(&spec, context).unwrap();
        assert_eq!(decoder.decode(0x1000, &[1]).unwrap().to_string(), "two");
        assert_eq!(
            FixedDecoder::with_context(&spec, ContextBytes::from_raw(vec![0; 9])).err(),
            Some(ContextError::InvalidLength {
                expected: 4,
                actual: 9
            })
        );
    }
}
