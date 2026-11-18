"""Replay of `holdtrack.analysis.where_lost` on the 57 annotated videos, base system, against the measurement of
the paper. Reads the annotations of the 57 videos. The replay data are described in docs/REPRODUCE.md.

    python analysis_where_lost.py <replay data> <work folder>
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from holdtrack.analysis import where_lost as wl

DATA, T = (os.path.abspath(x) for x in sys.argv[1:3])
SET = DATA + "/annotated57"
os.makedirs(T, exist_ok=True)
videos = sorted(os.listdir(SET + "/videos"))
print(len(videos), "videos")
assert len(videos) == 57
o = wl.measure(
    videos, DATA + "/dancetrack/{video}/gt/gt.txt", SET + "/videos/{video}/tracks.txt", None, 2
)
wl.show(o)
json.dump(o, open(f"{T}/report.json", "w"), indent=1)
ref = json.load(open(SET + "/reference/where_lost.json"))
e = max(abs(o["cases"][c] - ref[c]) for c in ref)
print(
    f"largest HOTA difference over the {len(ref)} combinations: {e:.2e} point | VERDICT:",
    "IDENTICAL" if e < 1e-6 else "DIFFERENT",
)
if e >= 1e-6:
    for c in ref:
        print(c, round(o["cases"][c], 4), round(ref[c], 4))
