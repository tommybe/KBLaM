import json

input_file = "nano_data/kblam_kb.json"         # JSON array input
output_file = "nano_data/kblam_kb_cleaned.jsonl"  # JSONL output

# Load the JSON array
with open(input_file, "r") as infile:
    data = json.load(infile)

# Clean and write to JSONL
with open(output_file, "w") as outfile:
    for entry in data:
        entry.pop("source_key", None)  # remove if exists
        outfile.write(json.dumps(entry) + "\n")

print(f"✅ Cleaned and saved {len(data)} entries to {output_file}")
