use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use clap::Parser;
use schemars::{schema_for, JsonSchema};
use serde::{Deserialize, Serialize};
use shared::types::{AssetMetadata, MaterialSpec, MessageHeader};

#[derive(JsonSchema, Serialize, Deserialize)]
struct SchemaRoot {
    asset: AssetMetadata,
    material: MaterialSpec,
    message: MessageHeader,
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Output file path for the JSON schema
    #[arg(short, long, value_name = "FILE")]
    output: PathBuf,
}

fn main() {
    let args = Args::parse();
    let schema = schema_for!(SchemaRoot);
    let json = serde_json::to_string_pretty(&schema).unwrap();

    let mut file = File::create(&args.output).expect("Failed to create output file");
    file.write_all(json.as_bytes()).expect("Failed to write to output file");
    
    println!("Schema successfully written to {:?}", args.output);
}

