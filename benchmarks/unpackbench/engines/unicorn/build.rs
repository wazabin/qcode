fn main() {
    // Unicorn's QEMU uses 16-byte atomics, which live in libatomic.
    println!("cargo:rustc-link-arg=-latomic");
}
