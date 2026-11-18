"""Fetch the models of HoldTrack into a weights folder and verify each file.

    python -m holdtrack.fetch_models --weights-dir weights            # light command
    python -m holdtrack.fetch_models --weights-dir weights --full     # complete system
    python -m holdtrack.fetch_models --weights-dir weights --mot17    # box corrector of MOT17

The HoldTrack weights come from the Hugging Face repository of the project. The other models are downloaded
from their authors: SAM 3 and SAM 3D Body (gated: accept their license on Hugging Face, then log in or set
HF_TOKEN), RTMW-L (MMPose). ViTPose+ is downloaded by `transformers` at the first run, and the two face models
by `holdtrack/faces/fetch_face_models.py`. Every file is checked against the SHA-256 of the file used for the
reported results; a file already present is only verified. Only the standard library is needed.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

HUB = "https://huggingface.co/{repo}/resolve/{revision}/{file}"
PROJECT = "TheophaneBourdon/HoldTrack"
RTMW_URL = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/"
    "rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.zip"
)
RTMW_ZIP_SHA = "a87e1af41a0a067776dba7d46e1c21c8f6e9f18e247e0e606718dd1f31e96ffd"

# destination in the weights folder: (repository, file in the repository, SHA-256)
LIGHT = {
    "rf_2xl.pth": (
        PROJECT,
        "rf_2xl.pth",
        "ccf2f57e34765514917bcea38158ad97b575e7070a4211e2dd91b1c72651d141",
    ),
    "sam3.pt": (
        "facebook/sam3",
        "sam3.pt",
        "9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e",
    ),
}
FULL = {
    "rf_large.pth": (
        PROJECT,
        "rf_large.pth",
        "8903b7d227cbbde936a42ba82cc47acca6d6bff8861248deabd7056f9a1f5f94",
    ),
    "yolo.pth": (
        PROJECT,
        "yolo.pth",
        "ee9cb04bf713a7c399bf7029eac2fce618075f010f64d34517ff0e989318e4bb",
    ),
    "dfine.pth": (
        PROJECT,
        "dfine.pth",
        "616f57e483c9ae66f020eb91d82577bffe3f3eccb81129f7c2a3dd63f3352709",
    ),
    "corrector/reference_trees.pkl": (
        PROJECT,
        "corrector/reference_trees.pkl",
        "8c5014e4986bb620404c24d3b2f726f7c49ce2b4e12ed4213ce432cc3aa8c2d5",
    ),
    "corrector/reference_network_0.pt": (
        PROJECT,
        "corrector/reference_network_0.pt",
        "7e08159e070667addbec5f0a8d9f5ba38afb3d93d390559620e636ef8c1d5ba2",
    ),
    "corrector/reference_network_1.pt": (
        PROJECT,
        "corrector/reference_network_1.pt",
        "c7072231894f3a300017c04b6d5ba66a82a38ab221a4663e0a265e0ea055e2ad",
    ),
    "corrector/trees.pkl": (
        PROJECT,
        "corrector/trees.pkl",
        "e348dfd82231960bb9364d310f6f815422c4e8c0bf66b930f069b161feb44a1f",
    ),
    "corrector/network_0.pt": (
        PROJECT,
        "corrector/network_0.pt",
        "3e8f4724af1cf0b64b5839cb8ea0f5292631d0c03673f4fe1981a4b88b5004ef",
    ),
    "corrector/network_1.pt": (
        PROJECT,
        "corrector/network_1.pt",
        "1c02c27450a913abc74d14b8227d06ac857f880a409f98fd5a3651cc43cb0a75",
    ),
    "corrector/pose_thresholds.json": (
        PROJECT,
        "corrector/pose_thresholds.json",
        "2783ad50b5ddf4b0c07b035403e963bf7ac105b69b70c0a65b7ed74ba574902b",
    ),
    "identity/appearance.pt": (
        PROJECT,
        "identity/appearance.pt",
        "3c9facef404b41303850340882b7b51054ebcce241d28bd752ee3dbc09b60e51",
    ),
    "sam3d_body/model.ckpt": (
        "facebook/sam-3d-body-dinov3",
        "model.ckpt",
        "b5a2f9d305dd02626b967aa2e86021fba07065df66ce7a7e00ffb9664f150abf",
    ),
    "sam3d_body/assets/mhr_model.pt": (
        "facebook/sam-3d-body-dinov3",
        "assets/mhr_model.pt",
        "352e271a6c42729c68554ceaea0c955e866970160c31e35506d782dc0f7377bc",
    ),
    "sam3d_body/model_config.yaml": (
        "facebook/sam-3d-body-dinov3",
        "model_config.yaml",
        "1012fc3f39cb5e90e3f8fbadf7bded31604bfafdce0321d17a7c1a2d3f08b88d",
    ),
}
MOT17 = {
    "mot17/trees.pkl": (
        PROJECT,
        "mot17/trees.pkl",
        "1e8e94913832117a76a543e1bb0ba8ce25695d1aca161d4ac725517112916fc1",
    ),
    "mot17/network_0.pt": (
        PROJECT,
        "mot17/network_0.pt",
        "2d793b89fb79f57fd18edaf1167061bc183727cc89fd69c60aea8ba7d0e46978",
    ),
    "mot17/network_1.pt": (
        PROJECT,
        "mot17/network_1.pt",
        "ea5d6c42ace5fd5ecb855c5a8c6f03cc0470d69588370c4d514c205d6987e95c",
    ),
    "mot17/network_2.pt": (
        PROJECT,
        "mot17/network_2.pt",
        "464ad713d792033de936729a9ab3f6a999783de1aec3a7ceb8d6ab235d5981e4",
    ),
    "mot17/network_3.pt": (
        PROJECT,
        "mot17/network_3.pt",
        "757ab5a44092fb47917e49fd842812bd4bdf1f966efbf78a2c923c271e542f5a",
    ),
}
RTMW = ("pose133.onnx", "bd033156e5104c4f5d2edfe0453e02661e30a2f3da453ec93c8764d561b83054")


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def token() -> str | None:
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"].strip()
    stored = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "token"
    return stored.read_text().strip() if stored.is_file() else None


def download(url: str, path: Path, authorization: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "holdtrack"}
    if authorization:
        headers["Authorization"] = "Bearer " + authorization
    partial = path.with_name(path.name + ".part")
    try:
        with (
            urllib.request.urlopen(urllib.request.Request(url, headers=headers)) as source,
            partial.open("wb") as target,
        ):
            while block := source.read(1 << 20):
                target.write(block)
    except urllib.error.HTTPError as error:
        if error.code in (401, 403):
            sys.exit(
                f"{url}: access refused ({error.code}). Accept the license of the model on "
                "Hugging Face, then run `huggingface-cli login` or set HF_TOKEN."
            )
        sys.exit(f"{url}: HTTP {error.code}")
    partial.replace(path)


def count_download(revision: str) -> None:
    """Request the config.json of the project repository, the file that Hugging Face counts as a download."""
    url = HUB.format(repo=PROJECT, revision=revision, file="config.json")
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "holdtrack"})
    try:
        urllib.request.urlopen(request, timeout=10).close()
    except (urllib.error.URLError, OSError):
        pass


def verify(path: Path, expected: str) -> None:
    got = sha256(path)
    if got != expected:
        sys.exit(f"{path}: SHA-256 {got}, expected {expected}")
    print("verified", path)


def expected_digests() -> dict[str, str]:
    from holdtrack.weights import manifest

    return {entry["filename"]: entry["sha256"] for entry in manifest().values()}


def fetch(weights: Path, full: bool = False, revision: str = "main", mot17: bool = False) -> None:
    digests = expected_digests()
    if mot17 and not full:
        files = dict(MOT17)
    else:
        files = dict(LIGHT, **(FULL if full else {}), **(MOT17 if mot17 else {}))
    counted = False
    for destination, (repository, file, digest) in files.items():
        path = weights / destination
        if not path.exists():
            print("downloading", destination, "from", repository)
            download(HUB.format(repo=repository, revision=revision, file=file), path, token())
            if repository == PROJECT and not counted:
                count_download(revision)
                counted = True
        verify(path, digest or digests[destination])
    if full:
        path = weights / RTMW[0]
        if not path.exists():
            archive = weights / "rtmw_l.zip"
            if not archive.exists():
                print("downloading", RTMW[0], "from MMPose")
                download(RTMW_URL, archive)
            verify(archive, RTMW_ZIP_SHA)
            with zipfile.ZipFile(archive) as z:
                member = next(n for n in z.namelist() if n.endswith("end2end.onnx"))
                with z.open(member) as source, path.open("wb") as target:
                    while block := source.read(1 << 20):
                        target.write(block)
        verify(path, RTMW[1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--weights-dir", default="weights")
    parser.add_argument(
        "--full", action="store_true", help="also the models of the complete system"
    )
    parser.add_argument(
        "--mot17", action="store_true", help="the box corrector of MOT17, alone or with --full"
    )
    parser.add_argument(
        "--revision", default="main", help="revision of the Hugging Face repositories"
    )
    options = parser.parse_args()
    fetch(Path(options.weights_dir).expanduser(), options.full, options.revision, options.mot17)


if __name__ == "__main__":
    main()
