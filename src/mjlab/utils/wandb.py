"""WandB utilities."""

from __future__ import annotations

import csv
import html
import math
import os
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence


def add_wandb_tags(tags: Sequence[str]) -> None:
  """Add tags to the current wandb run.

  Note: This function stores tags in wandb.config._wandb_tags if the run is not yet
  initialized, allowing them to be retrieved later. If the run is already initialized,
  tags are added directly.
  """
  if not tags:
    return

  try:
    import wandb

    if wandb.run is not None:
      existing_tags = list(wandb.run.tags) if wandb.run.tags else []
      new_tags = list(set(existing_tags + list(tags)))
      wandb.run.tags = new_tags
    else:
      # Store tags to be added when run is initialized.
      # This is a workaround for lazy wandb initialization in rsl_rl 3.1.0.
      current_tags = os.environ.get("WANDB_TAGS", "")
      all_tags = set(current_tags.split(",") if current_tags else [])
      all_tags.update(tags)
      os.environ["WANDB_TAGS"] = ",".join(sorted(all_tags))
  except ImportError:
    pass


def _safe_filename(name: str) -> str:
  safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")
  return safe_name or "metric"


@dataclass(frozen=True)
class WandbRunInfo:
  path: str
  run_id: str
  name: str
  start_time: float | None


def _as_finite_float(value: object) -> float | None:
  try:
    number = float(value)  # type: ignore[arg-type]
  except (TypeError, ValueError):
    return None
  return number if math.isfinite(number) else None


def _write_history_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
  columns: list[str] = []
  seen: set[str] = set()
  for row in rows:
    for key in row:
      if key not in seen:
        columns.append(key)
        seen.add(key)

  with path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)


def _metric_points(
  rows: Iterable[Mapping[str, object]], metric: str, x_key: str
) -> list[tuple[float, float]]:
  points: list[tuple[float, float]] = []
  for index, row in enumerate(rows):
    y = _as_finite_float(row.get(metric))
    if y is None:
      continue
    x = _as_finite_float(row.get(x_key))
    points.append((float(index) if x is None else x, y))
  return points


def _write_svg_line_plot(
  path: Path,
  *,
  title: str,
  x_label: str,
  y_label: str,
  points: Sequence[tuple[float, float]],
) -> None:
  width = 900
  height = 520
  margin_left = 88
  margin_right = 32
  margin_top = 58
  margin_bottom = 76
  plot_width = width - margin_left - margin_right
  plot_height = height - margin_top - margin_bottom

  xs = [point[0] for point in points]
  ys = [point[1] for point in points]
  x_min, x_max = min(xs), max(xs)
  y_min, y_max = min(ys), max(ys)

  if x_min == x_max:
    x_min -= 1.0
    x_max += 1.0
  if y_min == y_max:
    y_min -= 1.0
    y_max += 1.0

  def sx(x: float) -> float:
    return margin_left + ((x - x_min) / (x_max - x_min)) * plot_width

  def sy(y: float) -> float:
    return margin_top + plot_height - ((y - y_min) / (y_max - y_min)) * plot_height

  polyline = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in points)

  x_ticks = [x_min + (x_max - x_min) * i / 4 for i in range(5)]
  y_ticks = [y_min + (y_max - y_min) * i / 4 for i in range(5)]
  grid_lines = []
  for tick in x_ticks:
    x = sx(tick)
    grid_lines.append(
      f'<line x1="{x:.2f}" y1="{margin_top}" x2="{x:.2f}" '
      f'y2="{margin_top + plot_height}" stroke="#e5e7eb" />'
    )
    grid_lines.append(
      f'<text x="{x:.2f}" y="{height - 46}" text-anchor="middle">{tick:.4g}</text>'
    )
  for tick in y_ticks:
    y = sy(tick)
    grid_lines.append(
      f'<line x1="{margin_left}" y1="{y:.2f}" '
      f'x2="{margin_left + plot_width}" y2="{y:.2f}" stroke="#e5e7eb" />'
    )
    grid_lines.append(
      f'<text x="{margin_left - 12}" y="{y + 4:.2f}" '
      f'text-anchor="end">{tick:.4g}</text>'
    )

  title_text = html.escape(title)
  x_label_text = html.escape(x_label)
  y_label_text = html.escape(y_label)

  path.write_text(
    "\n".join(
      [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
          f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
          f'height="{height}" viewBox="0 0 {width} {height}">'
        ),
        '<rect width="100%" height="100%" fill="white" />',
        (
          "<style>text{font-family:Arial,sans-serif;font-size:13px;"
          "fill:#374151}.title{font-size:22px;font-weight:700;"
          "fill:#111827}.axis{font-size:15px;fill:#111827}</style>"
        ),
        (
          f'<text class="title" x="{width / 2}" y="32" '
          f'text-anchor="middle">{title_text}</text>'
        ),
        *grid_lines,
        (
          f'<line x1="{margin_left}" y1="{margin_top + plot_height}" '
          f'x2="{margin_left + plot_width}" y2="{margin_top + plot_height}" '
          'stroke="#111827" stroke-width="1.5" />'
        ),
        (
          f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" '
          f'y2="{margin_top + plot_height}" stroke="#111827" stroke-width="1.5" />'
        ),
        (
          f'<polyline points="{polyline}" fill="none" stroke="#2563eb" '
          'stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round" />'
        ),
        (
          f'<text class="axis" x="{width / 2}" y="{height - 14}" '
          f'text-anchor="middle">{x_label_text}</text>'
        ),
        (
          f'<text class="axis" transform="translate(20 {height / 2}) '
          f'rotate(-90)" text-anchor="middle">{y_label_text}</text>'
        ),
        "</svg>",
      ]
    ),
    encoding="utf-8",
  )


def get_current_wandb_run_info() -> WandbRunInfo | None:
  """Return identifying information for the active online W&B run."""
  try:
    import wandb
  except ImportError:
    return None

  if wandb.run is None:
    return None

  run_path_value = wandb.run.path
  if isinstance(run_path_value, (list, tuple)):
    run_path = "/".join(str(part) for part in run_path_value)
  else:
    run_path = str(run_path_value)
  run_id = wandb.run.id
  run_name = wandb.run.name or run_id
  created_at = wandb.run.start_time

  if not run_path or os.environ.get("WANDB_MODE") in {"disabled", "offline"}:
    return None

  return WandbRunInfo(
    path=run_path,
    run_id=run_id,
    name=run_name,
    start_time=created_at,
  )


def export_wandb_run_curves(
  run_info: WandbRunInfo, output_root: str | Path = "wandb/curves"
) -> Path | None:
  """Export a W&B run history to local CSV and SVG curve files."""
  try:
    import wandb
  except ImportError:
    return None

  api = wandb.Api()
  api_run = api.run(run_info.path)
  rows = list(api_run.scan_history())
  if not rows:
    return None

  date_prefix = ""
  if run_info.start_time is not None:
    date_prefix = datetime.fromtimestamp(run_info.start_time).strftime("%Y-%m-%d__")

  out_dir = Path(output_root) / _safe_filename(
    f"{date_prefix}{run_info.name}__{run_info.run_id}"
  )
  out_dir.mkdir(parents=True, exist_ok=True)
  _write_history_csv(out_dir / "history.csv", rows)

  x_key = "_step" if any("_step" in row for row in rows) else "_index"
  metric_keys: list[str] = []
  seen_keys: set[str] = set()
  for row in rows:
    for key in row:
      if key not in seen_keys:
        metric_keys.append(key)
        seen_keys.add(key)

  for key in metric_keys:
    if key.startswith("_"):
      continue
    points = _metric_points(rows, key, x_key)
    if len(points) < 2:
      continue
    _write_svg_line_plot(
      out_dir / f"{_safe_filename(key)}.svg",
      title=key,
      x_label=x_key,
      y_label=key,
      points=points,
    )

  return out_dir


def export_current_wandb_run_curves(
  output_root: str | Path = "wandb/curves",
) -> Path | None:
  """Finish and export the active W&B run to local CSV and SVG curve files.

  Returns the output directory when a run was exported, or ``None`` when no active
  online W&B run is available. Export failures are raised to the caller so training
  code can decide whether to warn or fail.
  """
  run_info = get_current_wandb_run_info()
  if run_info is None:
    return None

  import wandb

  # Flush the local process before reading the public history through the API.
  wandb.finish(quiet=True)

  return export_wandb_run_curves(run_info, output_root)
