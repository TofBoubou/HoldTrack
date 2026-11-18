# Analysis tools

These two measures apply to any multi-person tracker with annotated videos. The inputs are MOT text files (`frame, identity, x, y, width, height, ...`), one per video, and paths are given as patterns with `{video}`.

## Where a tracker loses HOTA

```
python -m holdtrack.analysis.where_lost --videos @videos.txt --gt "data/{video}/gt/gt.txt" --tracks "out/{video}.txt" --out report.json
```

The output of the tracker is corrected with the annotation in four ways, for wrong identities, misplaced boxes, missed persons and false boxes. HOTA is computed for the 16 combinations, and the gain of all four is split among them with the Shapley value, which does not depend on the order of the corrections. The same gain is also split by place, between corrections at contact, corrections away from contact and false boxes.

Output on our base system, for 57 annotated DanceTrack videos:

```
HOTA 83.065 ; missing 16.935 ; recovered by the four corrections +16.290 (HOTA 99.356)
one correction alone   : identity +4.644 ; box +6.089 ; missed +2.478 ; false +0.835
one correction last    : identity +6.350 ; box +7.589 ; missed +3.396 ; false +1.203
Shapley value          : identity +5.496 ; box +6.838 ; missed +2.940 ; false +1.017
by place, alone        : at contact +11.290 ; away from contact +2.678 ; false boxes +0.835
by place, Shapley      : at contact +11.994 ; away from contact +3.282 ; false boxes +1.014
```

`report.json` also holds the HOTA of every combination and the counts of boxes of each kind.

## Do several passes make the same identity errors?

```
python -m holdtrack.analysis.shared_errors --videos @videos.txt --gt "data/{video}/gt/gt.txt" --boxes "out/boxes/{video}.txt" \
    --pass native="out/native/{video}.txt" --pass exclusive="out/exclusive/{video}.txt" \
    --pass backward="out/backward/{video}.txt" --forward native,exclusive --backward backward
```

A pass is any run of a tracker on the video, with another setting, on the mirrored video or on the video played backward. The passes are compared on the boxes of `--boxes`, or on the annotated boxes if that option is absent. For each pair of passes, the report gives the errors of each pass, the boxes where both are wrong and the number expected if their errors were independent. Where both are wrong, it gives how often they name the same wrong person, with a 95% interval over videos and two chance levels. `--forward` and `--backward` add the comparison of pairs of forward passes with pairs made of a forward pass and the backward pass.

Output on our passes, for the same 57 videos (last line):

```
same wrong person: two forward passes 22.9 %, a forward and the backward pass 9.9 %, difference 13.0 points (9.3 to 17.8)
```

## Requirements

The tools need `numpy` and `scipy`. `where_lost` also needs TrackEval to compute HOTA (`pip install -e ".[eval]"`).
