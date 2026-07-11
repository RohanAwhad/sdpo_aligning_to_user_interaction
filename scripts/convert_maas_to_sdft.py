"""Convert maas_qna_with_chunks.jsonl to SDFT training format.

Input fields:  question, answer, context_chunk
Output fields: prompt, user_response, enriched_user_response

Usage:
  python scripts/convert_maas_to_sdft.py \
    --input tmp/maas_qna_with_chunks.jsonl \
    --output tmp/maas_sdft.jsonl
"""
import argparse
import json


def convert(input_path: str, output_path: str):
    with open(input_path) as f:
        records = [json.loads(line) for line in f]

    out = []
    for rec in records:
        out.append({
            "prompt": [{"from": "human", "value": rec["question"]}],
            "user_response": {"from": "human", "value": rec["answer"]},
            "enriched_user_response": {"from": "human", "value": rec["context_chunk"]},
        })

    with open(output_path, "w") as f:
        for rec in out:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Converted {len(out)} records: {input_path} -> {output_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    args = p.parse_args()
    convert(args.input, args.output)


if __name__ == "__main__":
    main()
