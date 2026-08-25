# wazabin-qcode-parser

Parser and AST for the QCode text format. Most users should depend on `qcode`
and use `qcode::lower::lower_str`; this crate is useful when a tool needs to
parse and inspect QCode syntax before lowering it.

It is also used by `wazabin-qcode-macro` to validate QCode literals at compile
time.
