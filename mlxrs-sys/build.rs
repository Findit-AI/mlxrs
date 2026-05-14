// Stub build script for Task 0.3.
// Task 0.4 will replace this with the real cmake + walkdir-driven build.
fn main() {
    println!("cargo:rerun-if-changed=build.rs");
}
