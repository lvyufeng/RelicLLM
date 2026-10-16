#!/usr/bin/env python3
"""Read what vLLM and SGLang claim about themselves, so a comparison page can be refreshed.

The tracking page `docs/architecture/upstream_capability_tracking.md` carries an upstream column
that is only true of the commit it was read at. This script reprints that column against any
checkout, the way `scripts/upstream_cli_inventory.py` reprints the command-line inventory behind
`docs/architecture/cli_surface_design_2026_09.md` -- it reads both projects as *text*, with no
import and no dependency on either being installed, so it re-runs against a future commit without a
build.

Three sources, because the two stacks publish their claims in different places:

* **vLLM** states them as two bullet lists in `README.md`, under ``vLLM is fast with:`` and
  ``vLLM is flexible and easy to use with:``.
* **SGLang's** `README.md` is an ecosystem page and carries no such list any more; its feature
  surface is the `docs/advanced_features/` directory, so the page names there are the inventory.
* **SGLang's** hardware support is a Markdown table in the README, and its first column is the list.

Usage::

    python scripts/upstream_capability_inventory.py --fetch
    python scripts/upstream_capability_inventory.py --vllm /tmp/vllm --sglang /tmp/sglang
    python scripts/upstream_capability_inventory.py --fetch --json

``--fetch`` reads the projects' public `main` branches over HTTP; a local checkout is preferred when
one is named, because it is the revision actually being discussed, and its HEAD and commit date are
recorded beside the reading. Both may be combined: a checkout per project where one exists.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import urllib.request
from datetime import date
from pathlib import Path

VLLM_README_URL = "https://raw.githubusercontent.com/vllm-project/vllm/main/README.md"
SGLANG_README_URL = "https://raw.githubusercontent.com/sgl-project/sglang/main/README.md"
SGLANG_SITEMAP_URL = "https://docs.sglang.io/sitemap.xml"

#: The SGLang feature pages live under this prefix. The sitemap is the directory listing, so the
#: slugs below it are the feature surface without cloning the repository.
SGLANG_FEATURE_PREFIX = "https://docs.sglang.io/docs/advanced_features/"

#: The two vLLM headings the capability list is cut by. Matched loosely because the sentence has
#: been reworded before ("vLLM is fast with:", "vLLM is flexible and easy to use with:").
_VLLM_HEADINGS = ("vllm is fast", "vllm is flexible")

_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_MD_EMPHASIS = re.compile(r"[*`]+")


class SourceError(RuntimeError):
    """A source could not be read, named so the caller can report which one."""


def _http_text(url: str, timeout: float = 30.0) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "relicllm-upstream-inventory"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def _read(path: Path, url: str, *, fetch: bool) -> str:
    """A file from ``path`` when it exists, else the HTTP copy when ``fetch`` is set."""
    if path.is_file():
        return path.read_text(encoding="utf-8", errors="replace")
    if fetch:
        return _http_text(url)
    raise SourceError(f"{path} is not a file and --fetch was not given")


def _plain(line: str) -> str:
    """A Markdown bullet as prose: links unwrapped, emphasis dropped, whitespace collapsed."""
    text = _MD_LINK.sub(r"\1", line)
    text = _MD_EMPHASIS.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def _bullets_under(text: str, headings: tuple[str, ...]) -> dict[str, list[str]]:
    """The bullet lists a document opens with a heading line matching one of ``headings``.

    A heading is any line whose lowercased text starts with the key; the bullets after it are
    collected until a non-blank line that is not a bullet. That is the shape both vLLM lists have,
    and it stops at the prose that follows them.
    """
    found: dict[str, list[str]] = {}
    lines = text.splitlines()
    current: str | None = None
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        lowered = line.lower()
        matched = next((key for key in headings if lowered.startswith(key)), None)
        if matched is not None and not line.startswith("-"):
            current = matched
            found.setdefault(current, [])
            continue
        if current is None:
            continue
        if line.startswith(("- ", "* ")):
            found[current].append(_plain(line[2:]))
        else:
            current = None
    return found


def _table_first_column(text: str, needle: str) -> list[str]:
    """The first-column labels of the Markdown table whose header row contains ``needle``.

    SGLang's hardware table is ``| Platform | Representative hardware |``; the labels under it are
    the platforms. Rows are read until the table ends.
    """
    labels: list[str] = []
    lines = text.splitlines()
    active = False
    for raw in lines:
        line = raw.strip()
        if not active:
            # The header is the first table row naming the column. It is matched only while the
            # table is not yet found: a *row* can contain the needle too -- SGLang's hardware rows
            # carry `hardware-platforms` in their links -- and re-matching there would swallow
            # every row as a second header.
            if line.startswith("|") and needle in line.lower():
                active = True
            continue
        if not line.startswith("|"):
            break
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if not cells or set(cells[0]) <= {"-", " ", ":"}:
            continue
        labels.append(_plain(cells[0]))
    return labels


def vllm_inventory(readme: str) -> dict[str, object]:
    return {"claims": _bullets_under(readme, _VLLM_HEADINGS)}


def sglang_inventory(readme: str, *, fetch: bool, sitemap: str | None) -> dict[str, object]:
    feature_pages: list[str] = []
    if sitemap:
        for url in re.findall(r"<loc>([^<]+)</loc>", sitemap):
            if not url.startswith(SGLANG_FEATURE_PREFIX):
                continue
            slug = url[len(SGLANG_FEATURE_PREFIX) :].strip("/")
            if slug and "/" not in slug:
                feature_pages.append(slug)
    return {
        "feature_pages": sorted(set(feature_pages)),
        "hardware": _table_first_column(readme, "platform"),
    }


def _checkout_ref(repo: Path) -> dict[str, str]:
    """HEAD and its commit date for a checkout, or an empty map when it is not a git tree."""
    try:
        head = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True, timeout=30,
        ).stdout.strip()
        when = subprocess.run(
            ["git", "-C", str(repo), "log", "-1", "--format=%cs"],
            capture_output=True, text=True, check=True, timeout=30,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return {}
    return {"head": head, "head_date": when}


def build(args: argparse.Namespace) -> dict[str, object]:
    inventory: dict[str, object] = {"read_at": date.today().isoformat(), "sources": {}}

    vllm_repo = Path(args.vllm) if args.vllm else None
    refs: dict[str, str] = {}
    if vllm_repo is not None:
        refs = _checkout_ref(vllm_repo)
    vllm_text = _read(
        (vllm_repo / "README.md") if vllm_repo else Path("/nonexistent/vllm/README.md"),
        VLLM_README_URL,
        fetch=args.fetch,
    )
    inventory["vllm"] = {"ref": refs, **vllm_inventory(vllm_text)}
    inventory["sources"]["vllm"] = str((vllm_repo / "README.md") if vllm_repo else VLLM_README_URL)

    sglang_repo = Path(args.sglang) if args.sglang else None
    refs = _checkout_ref(sglang_repo) if sglang_repo is not None else {}
    sglang_text = _read(
        (sglang_repo / "README.md") if sglang_repo else Path("/nonexistent/sglang/README.md"),
        SGLANG_README_URL,
        fetch=args.fetch,
    )
    sitemap: str | None = None
    local_features = (
        sglang_repo / "docs" / "advanced_features" if sglang_repo is not None else None
    )
    if local_features is not None and local_features.is_dir():
        slugs = sorted(p.stem for p in local_features.glob("*.md"))
        sitemap = "".join(
            f"<loc>{SGLANG_FEATURE_PREFIX}{slug}</loc>" for slug in slugs
        )
    elif args.fetch:
        try:
            sitemap = _http_text(SGLANG_SITEMAP_URL)
        except Exception as exc:  # noqa: BLE001 -- the README reading still stands without it
            print(f"warning: could not read the SGLang sitemap: {exc}", file=sys.stderr)
    inventory["sglang"] = {
        "ref": refs,
        **sglang_inventory(sglang_text, fetch=args.fetch, sitemap=sitemap),
    }
    inventory["sources"]["sglang"] = str(
        (sglang_repo / "README.md") if sglang_repo else SGLANG_README_URL
    )
    return inventory


def render(inventory: dict[str, object]) -> str:
    lines: list[str] = [f"Read at {inventory['read_at']}", ""]
    vllm = inventory["vllm"]  # type: ignore[index]
    lines.append("== vLLM ==")
    lines.append(f"    source: {inventory['sources']['vllm']}")  # type: ignore[index]
    if vllm.get("ref"):
        lines.append(f"    HEAD: {vllm['ref']['head']} ({vllm['ref']['head_date']})")
    for heading, bullets in vllm["claims"].items():
        lines.append(f"  [{heading}]")
        lines.extend(f"    - {item}" for item in bullets)
    lines.append("")
    sglang = inventory["sglang"]  # type: ignore[index]
    lines.append("== SGLang ==")
    lines.append(f"    source: {inventory['sources']['sglang']}")  # type: ignore[index]
    if sglang.get("ref"):
        lines.append(f"    HEAD: {sglang['ref']['head']} ({sglang['ref']['head_date']})")
    lines.append("  [advanced_features pages]")
    lines.extend(f"    - {item}" for item in sglang["feature_pages"])
    lines.append("  [hardware]")
    lines.extend(f"    - {item}" for item in sglang["hardware"])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--vllm", help="path to a vLLM checkout (its README.md is read)")
    parser.add_argument("--sglang", help="path to an SGLang checkout")
    parser.add_argument(
        "--fetch", action="store_true", help="read the public main branches over HTTP"
    )
    parser.add_argument("--json", action="store_true", help="emit JSON instead of text")
    args = parser.parse_args(argv)

    if not args.fetch and not (args.vllm or args.sglang):
        parser.error("give --fetch, or at least one of --vllm / --sglang")

    try:
        inventory = build(args)
    except SourceError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(inventory, indent=2, sort_keys=True))
    else:
        print(render(inventory))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())