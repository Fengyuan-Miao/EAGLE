import argparse
import glob
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from eagle.optim.metrics import read_jsonl, summarize_jsonl_records, write_summary_csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output", default="results/summary.csv")
    parser.add_argument("--ar-mode", default="vanilla_ar")
    args = parser.parse_args()

    paths = []
    for pattern in args.inputs:
        matches = glob.glob(pattern)
        paths.extend(matches if matches else [pattern])

    grouped = {}
    for path in paths:
        records = [record for record in read_jsonl(path) if record.get("record_type") in (None, "")]
        if not records:
            continue
        mode = records[0].get("mode", os.path.splitext(os.path.basename(path))[0])
        grouped.setdefault(mode, []).extend(records)

    ar_records = grouped.get(args.ar_mode, [])
    ar_tps = 0.0
    if ar_records:
        ar_tps = sum(float(r.get("tokens_per_s", 0.0)) for r in ar_records) / len(ar_records)

    rows = [summarize_jsonl_records(records, ar_tokens_per_s=ar_tps) for records in grouped.values()]
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    write_summary_csv(rows, args.output)


if __name__ == "__main__":
    main()
