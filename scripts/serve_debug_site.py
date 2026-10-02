#!/usr/bin/env python3
"""Serve a static debug site with gzip compression.

Use this instead of `python -m http.server` when debugging the sharded
dashboard locally. Python's built-in static server does not compress responses,
while Cloudflare Pages does, so this makes local transfer sizes closer to the
deployed site.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import mimetypes
import os
import shutil
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit


COMPRESSIBLE_SUFFIXES = {
    ".html",
    ".css",
    ".js",
    ".json",
    ".jsonl",
    ".txt",
    ".md",
    ".svg",
}

CACHE_DIR_NAME = ".gzip-cache"
SKIP_PRECOMPRESS_DIRS = {"docbuckets", "phrasebuckets", "topicphrases", CACHE_DIR_NAME}


class GzipStaticHandler(SimpleHTTPRequestHandler):
    server_version = "EviMapGzipHTTP/1.1"
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:
        self._serve(send_body=True)

    def do_HEAD(self) -> None:
        self._serve(send_body=False)

    def _serve(self, send_body: bool) -> None:
        path = self._resolve_path()
        if path is None:
            return

        content_type = self.guess_type(str(path))
        use_gzip = self._accepts_gzip() and path.suffix.lower() in COMPRESSIBLE_SUFFIXES
        if use_gzip:
            gz_path = gzip_cache_path(Path(self.directory).resolve(), path)
            ensure_gzip_cache(path, gz_path)
            data = gz_path.read_bytes()
        else:
            data = path.read_bytes()

        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Vary", "Accept-Encoding")
        if use_gzip:
            self.send_header("Content-Encoding", "gzip")
        self.end_headers()
        if send_body:
            self.wfile.write(data)

    def _resolve_path(self) -> Path | None:
        root = Path(self.directory).resolve()
        raw_path = unquote(urlsplit(self.path).path)
        rel = raw_path.lstrip("/")
        path = (root / rel).resolve()
        try:
            rel_path = path.relative_to(root)
        except ValueError:
            self.send_error(403, "Forbidden")
            return None
        if rel_path.parts and rel_path.parts[0] == CACHE_DIR_NAME:
            self.send_error(404, "File not found")
            return None

        if path.is_dir():
            index = path / "index.html"
            if index.exists():
                return index
            self.send_error(404, "Directory listing disabled")
            return None
        if not path.exists() or not path.is_file():
            self.send_error(404, "File not found")
            return None
        return path

    def _accepts_gzip(self) -> bool:
        enc = self.headers.get("Accept-Encoding", "")
        return "gzip" in {part.strip().split(";", 1)[0] for part in enc.split(",")}


def gzip_cache_path(root: Path, src: Path) -> Path:
    rel = src.relative_to(root)
    return root / CACHE_DIR_NAME / Path(str(rel) + ".gz")


def ensure_gzip_cache(src: Path, dst: Path) -> None:
    if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    with src.open("rb") as f_in, gzip.open(tmp, "wb", compresslevel=6) as f_out:
        shutil.copyfileobj(f_in, f_out, length=1024 * 1024)
    tmp.replace(dst)


def should_precompress(root: Path, path: Path, mode: str) -> bool:
    if path.suffix.lower() not in COMPRESSIBLE_SUFFIXES:
        return False
    rel = path.relative_to(root)
    if rel.parts and rel.parts[0] == CACHE_DIR_NAME:
        return False
    if mode == "all":
        return True
    if mode == "core":
        return not any(part in SKIP_PRECOMPRESS_DIRS for part in rel.parts)
    return False


def precompress(root: Path, mode: str, workers: int) -> None:
    if mode == "none":
        return
    files = [p for p in root.rglob("*") if p.is_file() and should_precompress(root, p, mode)]
    if not files:
        return
    print(
        f"Precompressing {len(files)} {mode} file(s) into {root / CACHE_DIR_NAME} "
        f"with {workers} worker(s) ...",
        flush=True,
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(ensure_gzip_cache, path, gzip_cache_path(root, path))
            for path in files
        ]
        for i, fut in enumerate(concurrent.futures.as_completed(futures), 1):
            fut.result()
            if i % 25 == 0 or i == len(files):
                print(f"  {i}/{len(files)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve site/debug with gzip compression.")
    parser.add_argument("--dir", default="site/debug", help="Directory to serve.")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind.")
    parser.add_argument("--bind", default="127.0.0.1", help="Address to bind.")
    parser.add_argument(
        "--precompress",
        choices=("none", "core", "all"),
        default="core",
        help="Prebuild gzip cache before serving. core excludes doc/phrase/topic buckets.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 4),
        help="Worker threads for precompression.",
    )
    args = parser.parse_args()

    root = Path(args.dir).resolve()
    if not root.exists():
        raise SystemExit(f"directory does not exist: {root}")

    mimetypes.add_type("application/json; charset=utf-8", ".json")
    mimetypes.add_type("application/json; charset=utf-8", ".jsonl")
    mimetypes.add_type("text/javascript; charset=utf-8", ".js")
    mimetypes.add_type("text/css; charset=utf-8", ".css")

    precompress(root, args.precompress, max(1, args.workers))

    handler = lambda *a, **kw: GzipStaticHandler(*a, directory=str(root), **kw)
    server = ThreadingHTTPServer((args.bind, args.port), handler)
    server.daemon_threads = True
    print(f"Serving {root} at http://{args.bind}:{args.port}/ with gzip compression")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
