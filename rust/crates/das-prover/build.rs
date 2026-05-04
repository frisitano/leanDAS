use std::path::PathBuf;
use std::process::Command;

fn main() {
    // Find the rec_aggregation crate's source directory via cargo metadata.
    // We need this path at compile time so we can locate its Python files
    // (recursion.py, whir.py, etc.) for symlink-based circuit compilation.
    let output = Command::new(std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_string()))
        .args(["metadata", "--format-version=1"])
        .output()
        .expect("failed to run cargo metadata");

    let stdout = String::from_utf8(output.stdout).expect("invalid utf8 from cargo metadata");

    // Simple JSON parsing: find "name":"rec_aggregation" and extract manifest_path.
    // We avoid a serde_json dependency by doing minimal string parsing.
    let rec_agg_manifest = find_manifest_path(&stdout, "rec_aggregation")
        .expect("could not find rec_aggregation in cargo metadata");

    let rec_agg_dir = PathBuf::from(&rec_agg_manifest)
        .parent()
        .expect("manifest_path has no parent")
        .to_str()
        .expect("non-utf8 path")
        .to_string();

    println!("cargo:rustc-env=REC_AGGREGATION_DIR={rec_agg_dir}");
    println!("cargo:rerun-if-changed=build.rs");
}

/// Extract the manifest_path for a package named `name` from cargo metadata JSON.
fn find_manifest_path(json: &str, name: &str) -> Option<String> {
    // Find the package-level entry: "name":"<name>","version":" distinguishes
    // package entries from dependency references (which lack "version").
    let name_pattern = format!("\"name\":\"{name}\",\"version\":\"");
    let idx = json.find(&name_pattern)?;
    let rest = &json[idx..];

    // manifest_path is in the same package object
    let mp_key = "\"manifest_path\":\"";
    let mp_start = rest.find(mp_key)? + mp_key.len();
    let mp_end = rest[mp_start..].find('"')? + mp_start;
    Some(rest[mp_start..mp_end].to_string())
}
