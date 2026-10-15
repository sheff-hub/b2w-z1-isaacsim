"""Open a stage inside the headless streaming app, at startup.

Passed to Kit with --exec by launch_headless_6.sh, which runs it once the app
itself is up. This is NOT the same job as open_stage_check.py: that one builds
its own SimulationApp to inspect a stage and exit, whereas this runs inside an
already-running streaming session so the result is visible over the livestream.

The stage path arrives in STAGE_TO_OPEN rather than as an --exec argument, so it
never has to survive shell quoting through `docker run -c "..."`.

Everything printed is prefixed OPENSTAGE| so it can be picked out of Kit's log.
"""

import os

import omni.usd

TAG = "OPENSTAGE|"


def say(*a):
    print(TAG, *a, flush=True)


path = os.environ.get("STAGE_TO_OPEN", "").strip()

if not path:
    say("STAGE_TO_OPEN is empty -- leaving the default empty stage")
elif not os.path.exists(path):
    # Worth distinguishing from a load failure: this one is a mount problem,
    # not a USD problem. The scenes root is mounted at /workspace/scenes.
    say(f"FATAL stage not found inside the container: {path}")
else:
    say(f"opening {path}")
    ctx = omni.usd.get_context()
    result = ctx.open_stage(path)
    say(f"open_stage returned {result!r}")

    stage = ctx.get_stage()
    if stage is None:
        say("FATAL open_stage produced no stage")
    else:
        say("root layer:", stage.GetRootLayer().identifier)
        say("default prim:", stage.GetDefaultPrim().GetName())
        say("prim count:", sum(1 for _ in stage.Traverse()))
        say("READY -- stage is live in the streaming session")
