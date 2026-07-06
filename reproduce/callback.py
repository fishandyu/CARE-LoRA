# reproduce/callback.py
import os
import json
from datetime import datetime
from transformers import TrainerCallback


def _ddp_is_main() -> bool:
    lr = os.environ.get("LOCAL_RANK")
    if lr is not None and str(lr).strip() != "":
        try:
            return int(lr) == 0
        except ValueError:
            return True
    rk = os.environ.get("RANK")
    if rk is not None and str(rk).strip() != "":
        try:
            return int(rk) == 0
        except ValueError:
            return True
    return True


class JsonlMetricsCallback(TrainerCallback):
    """Write Trainer log/eval metrics to a local JSONL file."""

    def __init__(self, jsonl_path: str, reset_on_start: bool = False, log_every_n_steps: int = 10):
        self.jsonl_path = jsonl_path
        self.log_every_n_steps = max(1, int(log_every_n_steps))
        parent = os.path.dirname(jsonl_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        if reset_on_start and os.path.exists(self.jsonl_path):
            os.remove(self.jsonl_path)

    def _append(self, payload: dict):
        payload["_time"] = datetime.now().isoformat(timespec="seconds")
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def on_log(self, args, state, control, logs=None, **kwargs):
        # Write local JSONL every N steps; in DDP only rank 0 writes the file.
        if (
            _ddp_is_main()
            and logs
            and (int(state.global_step) % self.log_every_n_steps == 0)
        ):
            self._append({"type": "log", "step": int(state.global_step), **logs})
        return control

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if _ddp_is_main() and metrics:
            self._append({"type": "eval", "step": int(state.global_step), **metrics})
        return control
