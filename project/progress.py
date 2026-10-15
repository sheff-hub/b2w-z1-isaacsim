"""Progress events of a fetch run, for ./process.sh in another terminal.

The simulator appends one JSON object per line to renders/progress/progress.jsonl (container path
/workspace/renders/progress/progress.jsonl):
  {"t": wall clock, "sim": sim s, "ev": name, ...details}
Event names: session, sim_ready, episode, phase (on every phase / pick-stage change), beat (every
2 s of sim time: pose, speed, distances), vln_done, nav2_plan, nav2_done, dock_done, pick_done,
deliver_done, result, summary, bye. Plain Python: no Isaac imports.
"""

import json
import os
import time

PATH = os.environ.get("FETCH_PROGRESS", "/workspace/renders/progress/progress.jsonl")


class Progress:
    def __init__(self, path=PATH, enabled=True):
        self.enabled = enabled
        self.path = path
        self.sim = 0.0
        self.last_beat = -1e9
        self.phase = None
        if enabled:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            open(path, "w").close()  # a new run starts a new file
            try:
                os.chmod(path, 0o666)
                os.chmod(os.path.dirname(path), 0o777)
            except OSError:
                pass

    def event(self, ev, **kv):
        if not self.enabled:
            return
        rec = {"t": round(time.time(), 2), "sim": round(self.sim, 2), "ev": ev}
        rec.update(kv)
        with open(self.path, "a") as f:
            f.write(json.dumps(rec, default=lambda o: round(float(o), 3)) + "\n")

    def tick(self, sim, phase, beat_fn=None, period=2.0):
        """Call every simulation step: emits 'phase' on a change and 'beat' every `period` s of sim time."""
        self.sim = sim
        if phase != self.phase:
            self.phase = phase
            self.event("phase", phase=phase)
        if beat_fn is not None and sim - self.last_beat >= period:
            self.last_beat = sim
            self.event("beat", phase=phase, **beat_fn())
