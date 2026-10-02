"""Download full paper archives; verify archive and member checksums before use."""
from __future__ import annotations

import argparse
import json
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path, PurePosixPath

from artifact_utils import sha256

ROOT = Path(__file__).resolve().parents[1]


def extract_verified(archive: Path, corpus: dict, root: Path) -> None:
    expected = {row["path"]: row for row in corpus["files"]}
    with tempfile.TemporaryDirectory(prefix="evimap-extract-") as temporary:
        stage = Path(temporary)
        with tarfile.open(archive, "r:gz") as handle:
            members = handle.getmembers()
            if len(members) != len(expected) or {m.name for m in members} != expected.keys():
                raise ValueError("archive member list differs from manifest")
            for member in members:
                path = PurePosixPath(member.name)
                if not member.isfile() or path.is_absolute() or ".." in path.parts:
                    raise ValueError(f"unsafe archive member: {member.name}")
                if member.size != expected[member.name]["bytes"]:
                    raise ValueError(f"member size mismatch: {member.name}")
                target = stage / member.name
                target.parent.mkdir(parents=True, exist_ok=True)
                with handle.extractfile(member) as source, target.open("wb") as destination:
                    shutil.copyfileobj(source, destination)
                if sha256(target) != expected[member.name]["sha256"]:
                    raise ValueError(f"member checksum mismatch: {member.name}")
        # Never silently replace an existing local run with downloaded data.
        for name in expected:
            target = root / name
            try:
                target.resolve().relative_to(root.resolve())
            except ValueError:
                raise ValueError(f"local symlink escapes checkout: {target}") from None
            if target.exists() and sha256(target) != expected[name]["sha256"]:
                raise FileExistsError(f"local file differs: {target}; choose a fresh checkout")
        for name in expected:
            target = root / name
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(stage / name, target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", action="append", help="select a corpus; default all six")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--archive-dir", type=Path, help="use already downloaded local archives")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "paper_artifacts/manifest.json").read_text())
    known = {c["id"] for c in manifest["corpora"]}
    if args.corpus and set(args.corpus) - known:
        parser.error("unknown corpus: " + ", ".join(set(args.corpus) - known))
    for corpus in manifest["corpora"]:
        if args.corpus and corpus["id"] not in args.corpus:
            continue
        if corpus["counts"]["documents"] < corpus["minimum_documents"]:
            raise ValueError("manifest corpus is smaller than the paper")
        with tempfile.TemporaryDirectory(prefix="evimap-download-") as temporary:
            archive = (args.archive_dir / corpus["archive"] if args.archive_dir else
                       Path(temporary) / corpus["archive"])
            if not args.archive_dir:
                url = manifest["release_base_url"] + "/" + corpus["archive"]
                print(f"[{corpus['id']}] downloading {corpus['archive_bytes']:,} bytes", flush=True)
                with urllib.request.urlopen(url, timeout=60) as response, archive.open("wb") as handle:
                    shutil.copyfileobj(response, handle)
            if archive.stat().st_size != corpus["archive_bytes"] or sha256(archive) != corpus["archive_sha256"]:
                raise ValueError(f"archive checksum/size mismatch: {archive}")
            extract_verified(archive, corpus, args.root.resolve())
        print(f"[{corpus['id']}] complete full cohort installed", flush=True)


if __name__ == "__main__":
    main()
