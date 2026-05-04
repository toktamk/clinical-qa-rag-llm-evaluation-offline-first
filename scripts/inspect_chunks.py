import json

path = "data/processed/chunks.jsonl"

with open(path, encoding="utf-8") as f:
    chunks = [json.loads(line) for line in f]

print("Total chunks:", len(chunks))

print("\nFirst chunk:")
print(json.dumps(chunks[0], indent=2, ensure_ascii=False))

print("\nShort chunks:")
for c in chunks:
    if c["token_count"] < 10:
        print(c["chunk_id"], "|", c["section"], "|", c["text"])