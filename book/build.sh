#!/usr/bin/env bash
# Builds the whole site into book/book: the reference from the crate, the
# book, the API docs under api/, and the playground's wasm under pkg/.
set -euo pipefail
cd "$(dirname "$0")/.."

cargo run -q -p qcode-langref > book/src/langref.md
(cd web && wasm-pack build --target web --release --out-dir pkg --no-typescript)
mkdir -p book/src/pkg
cp web/pkg/qcode_web.js web/pkg/qcode_web_bg.wasm book/src/pkg/
mdbook build book
cargo doc --workspace --no-deps
rm -rf book/book/api
cp -r target/doc book/book/api
