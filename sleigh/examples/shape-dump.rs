//! Prints the shape of one encoding and why each bit is decisive.
use sleigh::Decoder;
use sleigh::introspect::{DisassemblyAction, OperandKind};

fn main() {
    let spec = sleigh_precompile::x64::spec();
    let decoder = Decoder::new(spec);
    let dctx = spec.new_context();
    let view = spec.introspect();
    for hex in std::env::args().skip(1) {
        let bytes: Vec<u8> = (0..hex.len() / 2)
            .map(|k| u8::from_str_radix(&hex[2 * k..2 * k + 2], 16).unwrap())
            .collect();
        let (insn, shape) = decoder.decode_one_shaped(0x400000, &bytes, &dctx).unwrap();
        println!("{hex}: {}", insn.display().unwrap());
        println!("  mask   {:02x?}", shape.mask());
        println!("  params {:?}", shape.params());
        for m in insn.constructor_matches() {
            let table = view
                .tables()
                .find(|t| t.name() == m.table().name())
                .unwrap();
            let c = table.constructor(m.index());
            let ops: Vec<String> = c
                .operands()
                .iter()
                .map(|slot| match slot.kind {
                    OperandKind::Field(f) => {
                        let f = view.field(f).unwrap();
                        format!("field {}@{}{:?}", f.name(), slot.bit_offset, f.attachment())
                    }
                    OperandKind::Table(t) => format!(
                        "table {}@{}",
                        view.table(t).unwrap().name(),
                        slot.bit_offset
                    ),
                    OperandKind::Register(_) => "reg".into(),
                })
                .collect();
            let actions: Vec<String> = c
                .actions()
                .iter()
                .map(|a| match a {
                    DisassemblyAction::Assign { field, .. } => {
                        format!(
                            "assign {}",
                            view.field(*field).map(|f| f.name()).unwrap_or("?")
                        )
                    }
                    other => format!("{other:?}").chars().take(40).collect(),
                })
                .collect();
            let pats: Vec<String> = c
                .patterns()
                .map(|p| match p.instruction {
                    sleigh::introspect::PatternTest::Masked { mask, value } => {
                        format!("{mask:02x?}={value:02x?}")
                    }
                    other => format!("{other:?}"),
                })
                .collect();
            println!(
                "  {}[{}] ops {:?} actions {:?} patterns {:?}",
                m.table().name(),
                m.index(),
                ops,
                actions,
                pats
            );
        }
    }
}
