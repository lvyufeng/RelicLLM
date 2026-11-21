"""Verify sanitized Ascend publication and original manifest identities.

This is an offline file/metadata checker. It imports no model/runtime modules,
starts no tests or accelerator jobs, and never rewrites a recorded measurement.
Run from the repository root with ``python scripts/check_ascend_publication.py``.
"""
from __future__ import annotations

import hashlib
import html
import ipaddress
import json
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent.parent
EVIDENCE = ROOT / "docs/performance/evidence"
MAPPING = EVIDENCE / "ascend-publication-sanitization-20261121.json"
HOME = re.compile(r"/(?:Users|home)/(?!<|\$|\{)[^/\s\x22\x27<>`{}]+")
FLEET = re.compile(r"(?<![A-Za-z0-9])(?:ts-?|ascend)[0-9]{2,4}(?![A-Za-z0-9])", re.I)
WORKSTATION = re.compile(r"(?<![A-Za-z0-9])[A-Za-z0-9-]*Mac[A-Za-z0-9-]*\.local\b", re.I)
PRIVATE_ROOT = re.compile(r"/(?:data1|root)(?=/|[\s\x22\x27`]|$)")
IPV4 = re.compile(r"(?<![A-Za-z0-9_.])(?:[0-9]{1,3}\.){3}[0-9]{1,3}(?![A-Za-z0-9_.])")
IFACE = re.compile(
    r"(?<![A-Za-z0-9_])(?:eth[0-9]+|ib[0-9]+|hccn[0-9]+|bond[0-9]+|en[0-9]+|"
    r"en(?:p[0-9]+s[0-9]+(?:f[0-9]+)?(?:d[0-9]+)?|s[0-9]+(?:f[0-9]+)?(?:d[0-9]+)?|"
    r"o[0-9]+|x[0-9a-fA-F]{12}))(?![A-Za-z0-9_])"
)
MAC = re.compile(r"(?<![A-Za-z0-9:])(?:[0-9a-fA-F]{2}:){5}[0-9a-fA-F]{2}(?![A-Za-z0-9:])")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def checked_path(relative):
    path = ROOT / relative
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(ROOT):
        raise ValueError(f"publication identity must name a regular repository file: {relative}")
    return path


def load_json(data):
    def pairs(items):
        output = {}
        for key, value in items:
            if key in output:
                raise ValueError("duplicate JSON key in published metadata")
            output[key] = value
        return output
    return json.loads(data, object_pairs_hook=pairs)


def nonstandard_address(text):
    for match in IPV4.finditer(text):
        try:
            address = ipaddress.IPv4Address(match.group())
        except ipaddress.AddressValueError:
            continue
        if not address.is_loopback and str(address) not in {"0.0.0.0", "172.17.0.1"}:
            return True
    return False


def main():
    record = load_json(MAPPING.read_bytes())
    if record["schema_version"] != 1 or not record["measurement_hashes_and_sizes_not_recomputed"]:
        raise ValueError("publication mapping must preserve original measurement identities")
    entries = {row["path"]: row for row in record["files"]}
    if len(entries) != len(record["files"]) or len(entries) != record["files_changed"]:
        raise ValueError("publication mapping paths must be unique and complete")
    payloads = {}
    for relative, row in entries.items():
        data = checked_path(relative).read_bytes()
        if len(data) != row["published_bytes"] or digest(data) != row["published_sha256"]:
            raise ValueError(f"published bytes differ from their publication identity: {relative}")
        if not re.fullmatch(r"[0-9a-f]{64}", row["original_sha256"]) or row["original_bytes"] < 0:
            raise ValueError("invalid original raw identity in publication mapping")
        payloads[relative] = data

    manifests = 0
    retained = 0
    for directory in sorted(EVIDENCE.iterdir()):
        if not directory.is_dir() or not (directory / "manifest.json").is_file():
            continue
        manifest = load_json((directory / "manifest.json").read_bytes())
        files = manifest["files"]
        rows = files.items() if isinstance(files, dict) else ((row["path"], row) for row in files)
        for relative, raw_identity in rows:
            path = checked_path(directory.relative_to(ROOT) / relative)
            name = path.relative_to(ROOT).as_posix()
            data = payloads.get(name)
            if data is None:
                data = path.read_bytes()
            published = entries.get(name)
            original_size = published["original_bytes"] if published else len(data)
            original_sha = published["original_sha256"] if published else digest(data)
            if original_size != raw_identity["bytes"] or original_sha != raw_identity["sha256"]:
                raise ValueError(f"original measured identity was rewritten or lost: {name}")
            retained += 1
        manifests += 1

    names = subprocess.check_output([
        "git", "ls-files", "-z", "--", "docs/performance/evidence/", "docs/performance/ascend_*.md",
    ], cwd=ROOT).split(b"\0")
    paths = {Path(name.decode()) for name in names if name}
    paths.add(MAPPING.relative_to(ROOT))
    metadata = {"json": 0, "jsonl": 0, "xml": 0}
    for relative in sorted(paths):
        data = checked_path(relative).read_bytes()
        text = html.unescape(data.decode("utf-8", errors="replace"))
        if any(regex.search(text) for regex in (HOME, FLEET, WORKSTATION, PRIVATE_ROOT, IFACE, MAC)) or nonstandard_address(text):
            raise ValueError(f"private publication identifier remains: {relative}")
        if relative.suffix == ".json":
            load_json(data)
            metadata["json"] += 1
        elif relative.suffix == ".jsonl":
            for line in data.splitlines():
                if line.strip():
                    load_json(line)
            metadata["jsonl"] += 1
        elif relative.suffix == ".xml":
            ET.fromstring(data)
            metadata["xml"] += 1
    print(json.dumps({
        "publication_hashes_verified": len(entries),
        "original_manifests_verified": manifests,
        "original_retained_file_identities_verified": retained,
        "publication_files_scanned": len(paths),
        "private_identifier_residuals": 0,
        "structured_metadata_parsed": metadata,
        "tests_builds_benchmarks_and_accelerator_jobs_run": False,
    }, indent=2))


if __name__ == "__main__":
    main()
