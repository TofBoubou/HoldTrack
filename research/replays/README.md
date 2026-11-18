# Replays

The scripts that check the package against the frozen outputs of the system. Each one reads stored model
outputs (detector reads, tracker passes, poses, meshes, appearance stages, face tables) in a folder of replay
data, given as first argument; its layout is described in [docs/REPRODUCE.md](../../docs/REPRODUCE.md). They
run on a CPU. The face tables are not distributed; the two replays that read them need them computed again with
the face models of their authors.

| script | what it compares |
| --- | --- |
| `chain_test35.py` | from the stored model outputs to the final tracks of the 35 test videos, against the submission without faces |
| `faces_chain_test35.py` | the face steps, the relay, its check and the assembly on the 35 test videos, from a work folder of `chain_test35.py`, against the submission of the complete system |
| `identity_steps_test.py`, `identity_steps_annotated57.py` | the identity chain after each step, against the frozen identities |
| `identity_agree.py` | the agreement module alone |
| `faces_second_eye.py` | the second face model and the face tables, on 92 videos |
| `mot17_test.py` | the steps after the box corrector on the seven MOT17 test videos, against the files of the submission; data described in [docs/MOT17.md](../../docs/MOT17.md) |
| `analysis_where_lost.py`, `analysis_shared_errors.py` | two measurements of the paper |
