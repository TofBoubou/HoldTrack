"""Fetch the two face models of the optional face steps, from their own sources, and verify them.

    python fetch_face_models.py [--dest ~/.cache/holdtrack/faces]

ArcFace R100 (glintr100.onnx, InsightFace model pack "antelopev2") and AdaFace IR-101 WebFace12M
(Hugging Face, minchul/cvlface_adaface_ir101_webface12m, pinned revision). Both are released by their authors for
non-commercial research; HoldTrack does not redistribute them. Each file is checked against the SHA-256 of the
file used for the reported results. Only the Python standard library is needed.
"""

import argparse
import hashlib
import os
import sys
import urllib.request
import zipfile

ARCFACE_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/antelopev2.zip"
ARCFACE_SHA = {"glintr100.onnx": "4ab1d6435d639628a6f3e5008dd4f929edf4c4124b1a7169e1048f9fef534cdf"}
ADAFACE_REPO = "minchul/cvlface_adaface_ir101_webface12m"
ADAFACE_REV = "54f602a0737bd1ee4a4e7e9fd089a485f397fefd"
ADAFACE_SHA = {
    "model.safetensors": "2ea535a43877bd3de8091903935c783ce335be66a9f8917fae9a7a18ae4bbf56",
    "pretrained_model/model.pt": "e312d79222d28027f146ed495e182e48fe0dddf404cdbbdabcccdbdd07cc3758",
    "wrapper.py": "8676246c47090203367d757bd610ced89e002a59eaef29cd4be39e19b7d5181e",
    "config.json": "6311d95b23368f06932e6c04b52431c29857e1fcbf5e1d41a2e05f5e4b946d10",
    "files.txt": "1327f13e74b7306aee60fae49242d50150fbd7cf2f408fc111e6a3f4aa6917c1",
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(url, path):
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "holdtrack"})
    with urllib.request.urlopen(req) as r, open(path + ".part", "wb") as f:
        while True:
            block = r.read(1 << 20)
            if not block:
                break
            f.write(block)
    os.replace(path + ".part", path)


def check(path, expected):
    got = sha256(path)
    if got != expected:
        sys.exit(f"{path}: SHA-256 {got}, expected {expected}")
    print("verified", path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dest", default=os.path.expanduser("~/.cache/holdtrack/faces"))
    a = ap.parse_args()
    arc = os.path.join(a.dest, "arcface")
    ada = os.path.join(a.dest, "adaface")
    onnx = os.path.join(arc, "glintr100.onnx")
    if not os.path.exists(onnx):
        z = os.path.join(arc, "antelopev2.zip")
        download(ARCFACE_URL, z)
        with zipfile.ZipFile(z) as f:
            name = [n for n in f.namelist() if n.endswith("glintr100.onnx")][0]
            with f.open(name) as src, open(onnx, "wb") as dst:
                while True:
                    block = src.read(1 << 20)
                    if not block:
                        break
                    dst.write(block)
        os.remove(z)
    check(onnx, ARCFACE_SHA["glintr100.onnx"])
    base = f"https://huggingface.co/{ADAFACE_REPO}/resolve/{ADAFACE_REV}/"
    download(base + "files.txt", os.path.join(ada, "files.txt"))
    files = [lab for lab in open(os.path.join(ada, "files.txt")).read().split("\n") if lab] + [
        "config.json",
        "wrapper.py",
        "model.safetensors",
    ]
    for n in files:
        download(base + n, os.path.join(ada, n))
    for n, s in ADAFACE_SHA.items():
        check(os.path.join(ada, n), s)
    print("face models ready in", a.dest)


if __name__ == "__main__":
    main()
