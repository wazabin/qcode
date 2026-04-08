# qcode_analysis

Static analysis passes over QCode IR.

## Overview

Implements optimization and analysis passes including:
- dead code elimination (DCE)
- global value numbering (GVN)
- alias analysis
- and CFG simplification

Operates on the IR types defined in `qcode`.
