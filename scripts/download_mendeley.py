"""Download a public Mendeley Data dataset, keeping its folder structure.

Usage:
    python scripts/download_mendeley.py <dataset_id> <version> <out_dir> [--dry-run] [--workers 8]
"""
import argparse
import hashlib
import json
import os
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

API = "https://data.mendeley.com/public-api/datasets"


def get_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "dataset-downloader"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def list_files(ds, ver):
    folders = get_json(f"{API}/{ds}/folders/{ver}")
    by_id = {f["id"]: f for f in folders}

    def path_of(fid):
        parts = []
        while fid and fid in by_id:
            parts.append(by_id[fid]["name"].strip())
            fid = by_id[fid].get("parent_id")
        return os.path.join(*reversed(parts)) if parts else ""

    files = []
    for fid in ["root"] + list(by_id):
        for f in get_json(f"{API}/{ds}/files?folder_id={fid}&version={ver}"):
            rel = os.path.join(path_of(f.get("folder_id")), f["filename"])
            cd = f["content_details"]
            files.append((rel, cd["download_url"], f["size"], cd.get("sha256_hash")))
    return files


def download(item, out_dir):
    rel, url, size, sha = item
    dest = os.path.join(out_dir, rel)
    if os.path.exists(dest) and os.path.getsize(dest) == size:
        return "skip"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "dataset-downloader"})
    h = hashlib.sha256()
    tmp = dest + ".part"
    with urllib.request.urlopen(req, timeout=300) as r, open(tmp, "wb") as fh:
        while chunk := r.read(1 << 20):
            fh.write(chunk)
            h.update(chunk)
    if sha and h.hexdigest() != sha:
        os.remove(tmp)
        raise ValueError(f"hash mismatch: {rel}")
    os.replace(tmp, dest)
    return "ok"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dataset_id")
    p.add_argument("version", type=int)
    p.add_argument("out_dir")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()

    files = list_files(a.dataset_id, a.version)
    total = sum(f[2] for f in files)
    exts = {}
    for f in files:
        e = os.path.splitext(f[0])[1].lower()
        exts[e] = exts.get(e, 0) + 1
    print(f"{a.dataset_id} v{a.version}: {len(files)} files, {total / 1e9:.2f} GB, exts={exts}")
    if a.dry_run:
        return

    errors = 0
    with ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(download, f, a.out_dir): f for f in files}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                fut.result()
            except Exception as e:  # keep going, report at end
                errors += 1
                print("ERROR", futs[fut][0], e, file=sys.stderr)
            if i % 200 == 0:
                print(f"  {i}/{len(files)}", flush=True)
    print(f"done: {len(files) - errors} ok, {errors} errors")


if __name__ == "__main__":
    main()
