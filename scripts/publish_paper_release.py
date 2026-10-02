"""Publish verified full-cohort assets after the source commit is pushed."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from artifact_utils import sha256

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "zhiyintan/EviMap"


def credential() -> str:
    for key in ("GH_TOKEN", "GITHUB_TOKEN"):
        if os.environ.get(key):
            return os.environ[key]
    result = subprocess.run(["git", "credential", "fill"],
                            input="protocol=https\nhost=github.com\n\n",
                            text=True, capture_output=True,
                            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"})
    fields = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    if result.returncode or not fields.get("password"):
        raise RuntimeError("Configure GitHub credentials or set GH_TOKEN in your terminal.")
    return fields["password"]


def api(path: str, token: str, payload=None, method=None):
    request = urllib.request.Request(
        "https://api.github.com/" + path,
        data=None if payload is None else json.dumps(payload).encode(), method=method,
        headers={"Authorization": "Bearer " + token,
                 "Accept": "application/vnd.github+json",
                 "Content-Type": "application/json",
                 "X-GitHub-Api-Version": "2022-11-28"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


class ProgressReader:
    def __init__(self, handle, name):
        self.handle, self.name = handle, name
        self.total = self.last = 0

    def read(self, size):
        data = self.handle.read(size)
        self.total += len(data)
        if self.total - self.last >= 64 * 1024 * 1024:
            print(f"{self.name}: uploaded {self.total:,} bytes", flush=True)
            self.last = self.total
        return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, default=ROOT / "release_assets")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "paper_artifacts/manifest.json").read_text())
    for corpus in manifest["corpora"]:
        archive = args.assets / corpus["archive"]
        if archive.stat().st_size != corpus["archive_bytes"] or sha256(archive) != corpus["archive_sha256"]:
            raise ValueError(f"archive size/checksum mismatch: {archive}")
        if corpus["counts"]["documents"] < corpus["minimum_documents"]:
            raise ValueError("corpus is smaller than the paper")
        print(f"verified {archive.name}: {archive.stat().st_size:,} bytes", flush=True)
    if args.dry_run:
        return
    token = credential()
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    if status.strip():
        raise RuntimeError("Commit all source changes before publishing a release.")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    repo = api("repos/" + REPOSITORY, token)
    if not repo.get("permissions", {}).get("push"):
        raise PermissionError(f"GitHub identity lacks write access to {REPOSITORY}")
    branch = repo["default_branch"]
    reference = api(f"repos/{REPOSITORY}/git/ref/heads/{branch}", token)
    if reference["object"]["sha"] != commit:
        raise RuntimeError("Push this source commit to the repository's default branch first.")
    releases = api(f"repos/{REPOSITORY}/releases?per_page=100", token)
    release = next((row for row in releases if row["tag_name"] == manifest["release_tag"]), None)
    if release is None:
        release = api(f"repos/{REPOSITORY}/releases", token,
                      {"tag_name": manifest["release_tag"], "target_commitish": commit,
                       "name": "EviMap full paper artifacts v1", "draft": True,
                       "body": "Complete six-corpus artifacts accompanying arXiv:2609.06664v1. "
                       "203,444 normalized documents, full evidence spans, hierarchical maps, "
                       "and recorded baseline/diagnostic outputs. Archive and per-file SHA-256 "
                       "checksums are pinned in paper_artifacts/manifest.json. No corpus is "
                       "smaller than the paper cohort. Use scripts/fetch_paper_artifacts.py "
                       "to download and validate the datasets."})
    existing = {row["name"]: row for row in release["assets"]}
    upload_base = release["upload_url"].split("{", 1)[0]
    for corpus in manifest["corpora"]:
        name = corpus["archive"]
        previous = existing.get(name)
        expected_digest = "sha256:" + corpus["archive_sha256"]
        if previous:
            if previous.get("size") != corpus["archive_bytes"] or previous.get("digest") != expected_digest:
                raise RuntimeError(f"Existing release asset {name} differs; it has not been overwritten.")
            print(f"already uploaded and verified: {name}", flush=True)
            continue
        path = args.assets / name
        print(f"uploading {name}", flush=True)
        with path.open("rb") as handle:
            request = urllib.request.Request(
                upload_base + "?" + urllib.parse.urlencode({"name": name}),
                data=ProgressReader(handle, name), method="POST",
                headers={"Authorization": "Bearer " + token,
                         "Content-Type": "application/gzip",
                         "Content-Length": str(path.stat().st_size),
                         "Accept": "application/vnd.github+json"},
            )
            with urllib.request.urlopen(request, timeout=300) as response:
                uploaded = json.load(response)
        if uploaded.get("size") != corpus["archive_bytes"]:
            raise RuntimeError(f"GitHub asset size mismatch: {name}")
        if uploaded.get("digest") and uploaded["digest"] != expected_digest:
            raise RuntimeError(f"GitHub asset digest mismatch: {name}")
    if release["draft"]:
        release = api(f"repos/{REPOSITORY}/releases/{release['id']}", token,
                      {"draft": False}, method="PATCH")
    print(release["html_url"], flush=True)


if __name__ == "__main__":
    main()
