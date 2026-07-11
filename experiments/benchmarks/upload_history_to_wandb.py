from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.benchmarks.results import (
    DEFAULT_HISTORY_JSON,
    DEFAULT_WANDB_PROJECT,
    build_benchmark_records,
    log_benchmark_results_to_wandb,
)


def _parse_tags(tags: str) -> tuple[str, ...]:
    return tuple(tag.strip() for tag in tags.split(",") if tag.strip())


def load_history_rows(history_json: str) -> list[dict[str, Any]]:
    """Read benchmark history rows from an append-only JSON history file."""
    history_path = Path(history_json)
    rows = json.loads(history_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"History file must contain a JSON list: {history_path}")
    for row_idx, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"History row {row_idx} must be a JSON object.")
    return rows


def group_history_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group history rows by original benchmark run id while preserving order."""
    grouped: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for row_idx, row in enumerate(rows):
        run_id = str(row.get("run_id") or f"history-{row_idx:06d}")
        run_started_at = str(row.get("run_started_at") or "")
        command = row.get("command") if isinstance(row.get("command"), dict) else {}

        if run_id not in grouped:
            grouped[run_id] = {
                "run_id": run_id,
                "run_started_at": run_started_at,
                "command": command,
                "rows": [],
            }
        elif not grouped[run_id]["command"] and command:
            grouped[run_id]["command"] = command

        grouped[run_id]["rows"].append(row)
    return list(grouped.values())


def upload_history_to_wandb(
    *,
    history_json: str,
    project: str,
    entity: str = "",
    mode: str = "",
    tags: tuple[str, ...] = (),
    group_prefix: str = "",
    limit: int = 0,
    dry_run: bool = False,
) -> tuple[int, int]:
    """Upload existing local benchmark history rows to W&B."""
    rows = load_history_rows(history_json)
    groups = group_history_rows(rows)
    if limit > 0:
        groups = groups[:limit]

    total_rows = sum(len(group["rows"]) for group in groups)
    if dry_run:
        print(f"Would upload {total_rows} benchmark row(s) from {len(groups)} history run(s).")
        if groups:
            sample = groups[0]
            sample_records = build_benchmark_records(
                results=sample["rows"],
                command=sample["command"],
                run_id=sample["run_id"],
                run_started_at=sample["run_started_at"],
                extra_tags=tags,
            )
            if sample_records:
                print(json.dumps(sample_records[0], indent=2, sort_keys=True))
        return len(groups), total_rows

    for group in groups:
        wandb_group = f"{group_prefix}{group['run_id']}" if group_prefix else ""
        log_benchmark_results_to_wandb(
            results=group["rows"],
            command=group["command"],
            run_id=group["run_id"],
            run_started_at=group["run_started_at"],
            project=project,
            entity=entity,
            group=wandb_group,
            mode=mode,
            tags=tags,
        )
    print(f"Uploaded {total_rows} benchmark row(s) from {len(groups)} history run(s) to W&B project '{project}'.")
    return len(groups), total_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload existing Stream.FM benchmark history rows to W&B.")
    parser.add_argument("--history-json", default=DEFAULT_HISTORY_JSON)
    parser.add_argument("--wandb-project", default=DEFAULT_WANDB_PROJECT)
    parser.add_argument("--wandb-entity", default="")
    parser.add_argument("--wandb-mode", default="", help="Optional W&B mode, for example online, offline, or disabled.")
    parser.add_argument("--wandb-tags", default="", help="Comma-separated extra W&B tags.")
    parser.add_argument("--wandb-group-prefix", default="", help="Optional prefix for each history run_id W&B group.")
    parser.add_argument("--limit-runs", type=int, default=0, help="Upload only the first N history run groups. 0 uploads all.")
    parser.add_argument("--dry-run", action="store_true", help="Print the upload plan and one sample record without calling W&B.")
    args = parser.parse_args()

    upload_history_to_wandb(
        history_json=args.history_json,
        project=args.wandb_project,
        entity=args.wandb_entity,
        mode=args.wandb_mode,
        tags=_parse_tags(args.wandb_tags),
        group_prefix=args.wandb_group_prefix,
        limit=args.limit_runs,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
