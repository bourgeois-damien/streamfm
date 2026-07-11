from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.evaluation.results import (
    DEFAULT_EVAL_HISTORY_JSON,
    DEFAULT_EVAL_WANDB_PROJECT,
    build_eval_wandb_record,
    log_eval_result_to_wandb,
)


def _parse_tags(tags: str) -> tuple[str, ...]:
    return tuple(tag.strip() for tag in tags.split(",") if tag.strip())


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _score_result_for_eval_row(row: dict[str, Any]) -> dict[str, Any]:
    candidates = []
    summary_value = str(row.get("summary_path", ""))
    manifest_value = str(row.get("manifest_path", ""))
    if summary_value:
        summary_path = Path(summary_value)
        candidates.append(summary_path.with_name("metrics_all.json"))
    if manifest_value:
        manifest_path = Path(manifest_value)
        candidates.append(manifest_path.with_name("metrics_all.json"))
    for path in candidates:
        score_result = _read_json_if_exists(path)
        if score_result:
            return score_result
    return {}


def load_eval_history_rows(history_json: str) -> list[dict[str, Any]]:
    history_path = Path(history_json)
    rows = json.loads(history_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"History file must contain a JSON list: {history_path}")
    for row_idx, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"History row {row_idx} must be a JSON object.")
    return rows


def upload_eval_history_to_wandb(
    *,
    history_json: str,
    project: str,
    entity: str = "",
    mode: str = "",
    tags: tuple[str, ...] = (),
    group_prefix: str = "",
    limit: int = 0,
    include_scores: bool = True,
    dry_run: bool = False,
) -> int:
    rows = load_eval_history_rows(history_json)
    if limit > 0:
        rows = rows[:limit]

    if dry_run:
        print(f"Would upload {len(rows)} eval run(s).")
        if rows:
            row = rows[0]
            command = row.get("command") if isinstance(row.get("command"), dict) else {}
            score_result = _score_result_for_eval_row(row) if include_scores else {}
            sample = build_eval_wandb_record(
                result=row,
                command=command,
                score_result=score_result,
                extra_tags=tags,
            )
            print(json.dumps(sample, indent=2, sort_keys=True))
        return len(rows)

    for row in rows:
        command = row.get("command") if isinstance(row.get("command"), dict) else {}
        run_id = str(row.get("run_id") or command.get("run_name") or "")
        wandb_group = f"{group_prefix}{run_id}" if group_prefix and run_id else ""
        score_result = _score_result_for_eval_row(row) if include_scores else {}
        log_eval_result_to_wandb(
            result=row,
            command=command,
            score_result=score_result,
            project=project,
            entity=entity,
            group=wandb_group,
            mode=mode,
            tags=tags,
        )
    print(f"Uploaded {len(rows)} eval run(s) to W&B project '{project}'.")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload existing Stream.FM eval history rows to W&B.")
    parser.add_argument("--history-json", default=DEFAULT_EVAL_HISTORY_JSON)
    parser.add_argument("--wandb-project", default=DEFAULT_EVAL_WANDB_PROJECT)
    parser.add_argument("--wandb-entity", default="")
    parser.add_argument("--wandb-mode", default="", help="Optional W&B mode, for example online, offline, or disabled.")
    parser.add_argument("--wandb-tags", default="", help="Comma-separated extra W&B tags.")
    parser.add_argument("--wandb-group-prefix", default="", help="Optional prefix for each eval run_id W&B group.")
    parser.add_argument("--limit-runs", type=int, default=0, help="Upload only the first N eval runs. 0 uploads all.")
    parser.add_argument("--no-include-scores", action="store_true", help="Do not look for metrics_all.json next to eval summaries/manifests.")
    parser.add_argument("--dry-run", action="store_true", help="Print the upload plan and one sample record without calling W&B.")
    args = parser.parse_args()

    upload_eval_history_to_wandb(
        history_json=args.history_json,
        project=args.wandb_project,
        entity=args.wandb_entity,
        mode=args.wandb_mode,
        tags=_parse_tags(args.wandb_tags),
        group_prefix=args.wandb_group_prefix,
        limit=args.limit_runs,
        include_scores=not args.no_include_scores,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
