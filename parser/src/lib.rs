pub mod ast;
mod parser;

pub use parser::{ParseError, parse_program};

pub fn qcode_from_str(program: &str) -> Result<ast::Program, ParseError> {
    parse_program(program)
}
