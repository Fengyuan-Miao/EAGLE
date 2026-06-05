from eagle.optim.metrics import SUMMARY_FIELDS, summarize_jsonl_records


def test_metrics_schema_contains_required_fields():
    rows = [
        {
            "mode": "eagle_baseline",
            "tokens_per_s": 10.0,
            "wall_time_s": 2.0,
            "mean_accepted_len": 3.0,
            "mean_draft_depth": 5.0,
            "mean_draft_nodes": 32.0,
            "mean_verified_nodes": 32.0,
            "draft_time_ratio": 0.2,
            "verify_time_ratio": 0.7,
            "tree_construct_time_ratio": 0.1,
        }
    ]
    summary = summarize_jsonl_records(rows, ar_tokens_per_s=5.0)
    assert set(SUMMARY_FIELDS).issubset(summary)
    assert summary["mean_speedup_vs_ar"] == 2.0
