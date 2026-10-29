"""Fetch the official NASA C-MAPSS files (PHM Society S3 mirror of the NASA Prognostics Data Repository)."""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import requests

URLS = [
    "https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip",
    "https://data.nasa.gov/docs/legacy/CMAPSSData.zip",
]
WANTED = {f"{k}_FD00{i}.txt" for k in ("train", "test", "RUL") for i in range(1, 5)}


def have_all(raw_dir: Path) -> bool:
    return all((raw_dir / n).exists() for n in WANTED)


def _unpack(blob: bytes, raw_dir: Path) -> int:
    z = zipfile.ZipFile(io.BytesIO(blob))
    inner = [i for i in z.infolist() if i.filename.endswith("CMAPSSData.zip")]
    if inner:                                    # the mirror wraps the real archive in another zip
        z = zipfile.ZipFile(io.BytesIO(z.read(inner[0].filename)))
    n = 0
    for info in z.infolist():
        name = Path(info.filename).name         # flatten: defeats path traversal in archive member names
        if name in WANTED or name in ("readme.txt", "Damage Propagation Modeling.pdf"):
            (raw_dir / name).write_bytes(z.read(info))
            n += name in WANTED
    return n


def download(raw_dir: str | Path, force: bool = False) -> Path:
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    if have_all(raw_dir) and not force:
        return raw_dir
    last_err: Exception | None = None
    for url in URLS:
        try:
            r = requests.get(url, timeout=120)
            r.raise_for_status()
            if _unpack(r.content, raw_dir) == len(WANTED):
                return raw_dir
        except Exception as e:  # noqa: BLE001 - try the next mirror
            last_err = e
    raise RuntimeError(f"could not download C-MAPSS from any mirror ({last_err}). Place the 12 *_FD00x.txt files in {raw_dir}")
