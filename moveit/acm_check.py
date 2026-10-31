#!/usr/bin/env python3
"""Which link pairs of the Z1's collision shapes touch, over random joint states? (needs a running MoveIt)

    python3 moveit/acm_check.py [--n 2000]

MoveIt checks every link pair unless the SRDF disables it. Pairs in contact in (almost) every pose are an artefact
of the simplified shapes and must be disabled, or no plan is ever valid; the same goes for pairs touching in the
rest pose the robot starts every pick from (all joints 0, the folded arm). Prints those pairs as SRDF lines.
"""
import argparse
import json
import random
import urllib.request
from collections import Counter

LIMITS = [(-2.617994, 2.617994), (0.0, 2.967060), (-2.879793, 0.0), (-1.518436, 1.518436), (-1.343904, 1.343904),
          (-2.792527, 2.792527)]


def check(port, q, g=0.0):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/check", data=json.dumps({"state": q, "gripper": g}).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=30).read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--port", type=int, default=5810)
    a = ap.parse_args()
    rng = random.Random(0)
    count = Counter()
    for _ in range(a.n):
        q = [rng.uniform(lo, hi) for lo, hi in LIMITS]
        for pair in check(a.port, q, rng.uniform(-1.5708, 0.0))["contacts"]:
            count[tuple(pair)] += 1
    rest = {tuple(p) for g in (0.0, -1.5708) for p in check(a.port, [0.0] * 6, g)["contacts"]}
    print(f"{a.n} random states; pair: share of states in contact")
    for pair, k in count.most_common():
        print(f"  {pair[0]:14s} {pair[1]:14s} {k / a.n:6.1%}{'   (rest pose)' if pair in rest else ''}")
    print("\nSRDF lines (always in contact > 95 %, or touching in the rest pose):")
    for pair in sorted(rest | {p for p, k in count.items() if k / a.n > 0.95}):
        why = "Default" if pair in rest else "Always"
        print(f'  <disable_collisions link1="{pair[0]}" link2="{pair[1]}" reason="{why}"/>')


if __name__ == "__main__":
    main()
