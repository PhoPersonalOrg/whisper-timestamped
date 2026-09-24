"""
Delete Finder/export-style duplicate .m4a copies named like:
  UUID.m4a
  UUID 2.m4a
  UUID 3.m4a

For each base stem, keep the largest file (larger size may mean more recorded
content). On a size tie, prefer the unsuffixed original, then the lowest
export number. If the keeper is a numbered copy, rename it to the unsuffixed
name after deleting the others.

Dry-run by default; pass --execute to apply deletes/renames.
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import DefaultDict, Dict, List, Optional, Tuple

# Trailing " 2", " 3", ... before the extension (macOS/Windows copy naming).
_EXPORT_SUFFIX_RE = re.compile(r"^(?P<base>.+?)(?: (?P<n>\d+))?$", re.IGNORECASE)

DEFAULT_AUDIO_DIR = Path(r"H:\backups\2026-09-21_iPhone15Pro\WhisperApp\Audio")


@dataclass(frozen=True)
class AudioExport:
    path: Path
    base: str
    export_n: int  # 0 = unsuffixed original
    size_bytes: int


def parse_export_stem(stem: str) -> Optional[Tuple[str, int]]:
    """Return (base, export_n) or None if the stem does not match."""
    match = _EXPORT_SUFFIX_RE.match(stem)
    if not match:
        return None
    base = match.group("base")
    n_raw = match.group("n")
    export_n = int(n_raw) if n_raw is not None else 0
    return base, export_n


def collect_m4a_exports(audio_dir: Path) -> List[AudioExport]:
    exports: List[AudioExport] = []
    for path in sorted(audio_dir.glob("*.m4a")):
        if not path.is_file():
            continue
        parsed = parse_export_stem(path.stem)
        if parsed is None:
            print(f"  ! Skipping unrecognized name: {path.name}")
            continue
        base, export_n = parsed
        exports.append(
            AudioExport(
                path=path,
                base=base,
                export_n=export_n,
                size_bytes=path.stat().st_size,
            )
        )
    ## END for path in sorted(audio_dir.glob("*.m4a"))....

    return exports


def group_by_base(exports: List[AudioExport]) -> Dict[str, List[AudioExport]]:
    groups: DefaultDict[str, List[AudioExport]] = defaultdict(list)
    for export in exports:
        groups[export.base].append(export)
    ## END for export in exports....

    return dict(groups)


def choose_keeper(group: List[AudioExport]) -> AudioExport:
    """Largest size wins; ties prefer unsuffixed (n=0), then lowest n."""
    return max(group, key=lambda e: (e.size_bytes, -e.export_n))


def plan_dedupe(
    groups: Dict[str, List[AudioExport]],
) -> Tuple[List[Tuple[str, AudioExport, List[AudioExport], Optional[Path]]], int]:
    """
    Returns (plans, singleton_count) where each plan is:
      (base, keeper, to_delete, rename_to_or_None)
    """
    plans: List[Tuple[str, AudioExport, List[AudioExport], Optional[Path]]] = []
    singleton_count = 0

    for base, group in sorted(groups.items()):
        if len(group) < 2:
            singleton_count += 1
            continue

        keeper = choose_keeper(group)
        to_delete = [e for e in group if e.path != keeper.path]
        rename_to: Optional[Path] = None
        if keeper.export_n != 0:
            # Prefer canonical unsuffixed name after removing smaller copies.
            rename_to = keeper.path.with_name(f"{base}{keeper.path.suffix}")

        plans.append((base, keeper, to_delete, rename_to))
    ## END for base, group in sorted(groups.items())....

    return plans, singleton_count


def format_size(n: int) -> str:
    if n >= 1_048_576:
        return f"{n / 1_048_576:.3f} MB"
    if n >= 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n} B"


def run_dedupe(audio_dir: Path, execute: bool) -> None:
    audio_dir = audio_dir.resolve()
    if not audio_dir.is_dir():
        raise SystemExit(f"Audio directory not found: {audio_dir}")

    mode = "EXECUTE" if execute else "DRY-RUN"
    print(f"[{mode}] Scanning: {audio_dir}")

    exports = collect_m4a_exports(audio_dir)
    groups = group_by_base(exports)
    plans, singleton_count = plan_dedupe(groups)

    delete_count = sum(len(to_delete) for _, _, to_delete, _ in plans)
    rename_count = sum(1 for _, _, _, rename_to in plans if rename_to is not None)
    size_diff_groups = 0

    print(f"Found {len(exports)} .m4a file(s) in {len(groups)} base group(s).")
    print(f"  Unique (no duplicates): {singleton_count}")
    print(f"  Duplicate groups:       {len(plans)}")
    print(f"  Files to delete:        {delete_count}")
    print(f"  Keepers to rename:      {rename_count}")
    print()

    for base, keeper, to_delete, rename_to in plans:
        sizes = {e.size_bytes for e in [keeper, *to_delete]}
        if len(sizes) > 1:
            size_diff_groups += 1
            print(f"* {base}  (sizes differ — keeping largest)")
        else:
            print(f"* {base}")

        print(
            f"    KEEP   {keeper.path.name}  "
            f"({format_size(keeper.size_bytes)}, export_n={keeper.export_n})"
        )
        for doomed in sorted(to_delete, key=lambda e: e.export_n):
            print(
                f"    DELETE {doomed.path.name}  "
                f"({format_size(doomed.size_bytes)}, export_n={doomed.export_n})"
            )
        ## END for doomed in sorted(to_delete, key=lambda e: e.export_n)....

        if rename_to is not None:
            print(f"    RENAME {keeper.path.name} -> {rename_to.name}")

        if execute:
            for doomed in to_delete:
                doomed.path.unlink()
                print(f"      deleted {doomed.path.name}")
            ## END for doomed in to_delete....

            if rename_to is not None:
                if rename_to.exists():
                    raise RuntimeError(
                        f"Cannot rename {keeper.path.name} -> {rename_to.name}: target exists"
                    )
                keeper.path.rename(rename_to)
                print(f"      renamed -> {rename_to.name}")
    ## END for base, keeper, to_delete, rename_to in plans....

    print()
    print(f"Groups with unequal sizes: {size_diff_groups}")
    if not execute:
        print("Dry-run only. Re-run with --execute to delete/rename.")
    else:
        print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Delete duplicate .m4a export copies ( name 2 /  3 / ...), keeping the largest."
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=DEFAULT_AUDIO_DIR,
        help=f"Directory of .m4a files (default: {DEFAULT_AUDIO_DIR})",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually delete/rename files (default is dry-run)",
    )
    args = parser.parse_args()
    run_dedupe(audio_dir=args.audio_dir, execute=args.execute)


if __name__ == "__main__":
    main()
