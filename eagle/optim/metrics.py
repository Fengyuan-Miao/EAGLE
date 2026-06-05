import csv
import json
import statistics
from typing import Dict, Iterable, List


SUMMARY_FIELDS = [
    "mode",
    "num_prompts",
    "mean_tokens_per_s",
    "mean_speedup_vs_ar",
    "mean_accepted_len",
    "mean_draft_depth",
    "mean_draft_nodes",
    "mean_verified_nodes",
    "draft_time_ratio",
    "verify_time_ratio",
    "tree_construct_time_ratio",
    "p50_latency_s",
    "p95_latency_s",
]


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _quantile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = min(len(values) - 1, int(round((len(values) - 1) * q)))
    return values[idx]


def summarize_jsonl_records(records: List[Dict], ar_tokens_per_s: float = 0.0) -> Dict:
    if not records:
        return {field: 0 for field in SUMMARY_FIELDS}
    mode = records[0].get("mode", "")
    tokens_per_s = [float(r.get("tokens_per_s", 0.0)) for r in records]
    latencies = [float(r.get("wall_time_s", 0.0)) for r in records]
    mean_tps = _mean(tokens_per_s)
    return {
        "mode": mode,
        "num_prompts": len(records),
        "mean_tokens_per_s": mean_tps,
        "mean_speedup_vs_ar": mean_tps / ar_tokens_per_s if ar_tokens_per_s else 0.0,
        "mean_accepted_len": _mean(float(r.get("mean_accepted_len", 0.0)) for r in records),
        "mean_draft_depth": _mean(float(r.get("mean_draft_depth", 0.0)) for r in records),
        "mean_draft_nodes": _mean(float(r.get("mean_draft_nodes", 0.0)) for r in records),
        "mean_verified_nodes": _mean(float(r.get("mean_verified_nodes", 0.0)) for r in records),
        "draft_time_ratio": _mean(float(r.get("draft_time_ratio", 0.0)) for r in records),
        "verify_time_ratio": _mean(float(r.get("verify_time_ratio", 0.0)) for r in records),
        "tree_construct_time_ratio": _mean(float(r.get("tree_construct_time_ratio", 0.0)) for r in records),
        "p50_latency_s": statistics.median(latencies) if latencies else 0.0,
        "p95_latency_s": _quantile(latencies, 0.95),
    }


def read_jsonl(path: str) -> List[Dict]:
    records = []
    with open(path) as fin:
        for line in fin:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_summary_csv(rows: List[Dict], output: str):
    with open(output, "w", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, 0) for field in SUMMARY_FIELDS})
