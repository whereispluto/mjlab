import sys
from types import SimpleNamespace

from mjlab.utils.wandb import export_current_wandb_run_curves


def test_export_current_wandb_run_curves_writes_csv_and_svgs(tmp_path, monkeypatch):
  rows = [
    {"_step": 0, "train/loss": 1.0, "episode reward": 0.5},
    {"_step": 1, "train/loss": 0.75, "episode reward": 1.25},
    {"_step": 2, "train/loss": 0.5, "episode reward": 2.0},
  ]

  class FakeApiRun:
    def scan_history(self):
      return rows

  class FakeApi:
    def run(self, run_path):
      assert run_path == "entity/project/run123"
      return FakeApiRun()

  fake_wandb = SimpleNamespace(
    Api=FakeApi,
    finish=lambda quiet: None,
    run=SimpleNamespace(
      path=["entity", "project", "run123"],
      id="run123",
      name="example run",
      start_time=None,
    ),
  )
  monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

  out_dir = export_current_wandb_run_curves(tmp_path)

  assert out_dir is not None
  assert (out_dir / "history.csv").exists()
  assert (out_dir / "train_loss.svg").exists()
  assert (out_dir / "episode_reward.svg").exists()
