//! Passes related to CFG discovery and manipulation.

mod cfg_simplify;
mod crt;
mod jump_table;

pub use cfg_simplify::{SimplifyCfg, absorb_straight_line, simplify_cfg};
pub use crt::DiscoverLibcMain;
pub use jump_table::HandleJumpTables;
