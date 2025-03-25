import json
import os
import openai
from tqdm import tqdm

# Load OpenAI API key (must be set in your environment)
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Input file
INPUT_FILE = "nano_data/final_analysis_case_studies_fixed.json"
OUTPUT_FILE = "nano_data/kblam_kb.json"
TEMP_FILE = "nano_data/kblam_kb_output_temp.jsonl"

# ChatGPT Model
MODEL = "o3-mini"

# Load your JSON
with open(INPUT_FILE, "r") as f:
    case_data = json.load(f)

# GPT system prompt to guide consistent triple formatting
SYSTEM_PROMPT = """
You are an expert assistant designed to extract structured knowledge base triples for the KBLam model.
Given a technical case description, extract a list of dictionaries where each dictionary has:
- name: entity name (e.g., machine or component)
- description_type: one of ['description', 'symptoms', 'diagnosis', 'actions', 'outcome']
- description: textual content for that type
- Q: a natural language question related to the triple
- A: answer based on description and entity
- key_string: "the {description_type} of {name}"

Only include relevant triples. Do NOT include anything if the input says 'Not relevant'.
Return ONLY a valid JSON array of dictionaries.
"""

# Load already processed keys if resuming
processed_keys = set()
if os.path.exists(TEMP_FILE):
    with open(TEMP_FILE, "r") as f:
        for line in f:
            try:
                entry = json.loads(line)
                processed_keys.add(entry.get("source_key"))
            except:
                continue

def extract_triples_with_gpt(case_key, case_text):
    if "not relevant" in case_text.lower():
        return []

    user_prompt = f"""Extract structured triples from the following case:\n\n{case_text}"""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
        )
        content = response.choices[0].message.content.strip()
        triples = json.loads(content)
        return triples
    except Exception as e:
        print(f"❌ Error processing {case_key}: {e}")
        return []

# Main loop with incremental save
with open(TEMP_FILE, "a") as temp_out:
    for filename, content in tqdm(case_data.items(), desc="Processing cases"):
        if filename in processed_keys or "not relevant" in content.strip().lower():
            continue

        triples = extract_triples_with_gpt(filename, content)

        for triple in triples:
            triple["source_key"] = filename  # optional for traceability
            temp_out.write(json.dumps(triple) + "\n")
            temp_out.flush()  # ensure it's saved

# Combine and deduplicate results
final_output = []
seen = set()
with open(TEMP_FILE, "r") as f:
    for line in f:
        try:
            entry = json.loads(line)
            key = (entry.get("key_string"), entry.get("source_key"))
            if key not in seen:
                seen.add(key)
                final_output.append(entry)
        except:
            continue

# Write cleaned output
with open(OUTPUT_FILE, "w") as f:
    json.dump(final_output, f, indent=2)

print(f"\n✅ Saved {len(final_output)} unique triples to {OUTPUT_FILE}")