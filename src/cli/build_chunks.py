import argparse
from pathlib import Path

from src.chunking.section_chunker import chunk_files, write_jsonl, validate_chunks


def build_chunks(input_dir: str, output_path: str):
    input_dir = Path(input_dir)
    paths = sorted(input_dir.glob("*.txt"))

    chunks = chunk_files(paths)
    report = validate_chunks(chunks)

    if report:
        print("Chunk validation warnings:")
        for chunk_id, warnings in report.items():
            print(f"- {chunk_id}: {warnings}")

    write_jsonl(chunks, output_path)
    print(f"Exported {len(chunks)} chunks to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    build_chunks(args.input_dir, args.output)