import json
import os
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterable, Optional

import torch


class StepProfiler:
    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self.records = []
        self._current = None

    def reset(self):
        self.records = []
        self._current = None

    def start_step(self, step: int):
        if not self.enabled:
            return
        self._current = {"step": int(step)}

    def finish_step(self):
        if not self.enabled or self._current is None:
            return
        self.records.append(self._current)
        self._current = None

    @contextmanager
    def time(self, name: str):
        if not self.enabled:
            yield
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        try:
            yield
        finally:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            key = f"{name}_time_ms"
            if self._current is not None:
                self._current[key] = self._current.get(key, 0.0) + elapsed_ms

    def add_scalar(self, name: str, value: Any):
        if not self.enabled or self._current is None:
            return
        if isinstance(value, torch.Tensor):
            value = value.item()
        self._current[name] = value

    def summarize(self) -> Dict[str, Any]:
        if not self.records:
            return {}

        def mean(name: str) -> float:
            vals = [float(r[name]) for r in self.records if name in r]
            return sum(vals) / len(vals) if vals else 0.0

        summary = {
            "num_steps": len(self.records),
            "mean_accepted_len": mean("accepted_len"),
            "mean_draft_nodes": mean("draft_nodes"),
            "mean_verified_nodes": mean("verified_nodes"),
            "mean_draft_depth": mean("actual_draft_depth"),
        }
        time_keys = [
            "draft_time_ms",
            "tree_construct_time_ms",
            "verify_time_ms",
            "accept_time_ms",
            "sampling_time_ms",
            "step_time_ms",
        ]
        for key in time_keys:
            summary[f"mean_{key}"] = mean(key)

        total_time = sum(
            float(r.get("step_time_ms", 0.0))
            or sum(float(r.get(k, 0.0)) for k in time_keys if k != "step_time_ms")
            for r in self.records
        )
        if total_time > 0:
            for key in ("draft_time_ms", "verify_time_ms", "tree_construct_time_ms", "accept_time_ms"):
                summary[key.replace("_time_ms", "_time_ratio")] = (
                    sum(float(r.get(key, 0.0)) for r in self.records) / total_time
                )
        return summary

    def iter_jsonl_records(self, metadata: Optional[Dict[str, Any]] = None) -> Iterable[Dict[str, Any]]:
        metadata = metadata or {}
        for record in self.records:
            yield {**metadata, "record_type": "step", **record}
        yield {**metadata, "record_type": "summary", **self.summarize()}

    def write_jsonl(self, path: str, metadata: Optional[Dict[str, Any]] = None, append: bool = True):
        if not self.enabled:
            return
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        mode = "a" if append else "w"
        with open(path, mode) as fout:
            for record in self.iter_jsonl_records(metadata):
                fout.write(json.dumps(record) + "\n")
