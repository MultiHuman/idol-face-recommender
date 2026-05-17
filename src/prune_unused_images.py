from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


@dataclass(frozen=True)
class PrunePlan:
    raw_files: list[Path]
    crop_files: list[Path]

    @property
    def file_count(self) -> int:
        return len(self.raw_files) + len(self.crop_files)

    @property
    def byte_count(self) -> int:
        total = 0
        for path in [*self.raw_files, *self.crop_files]:
            try:
                total += path.stat().st_size
            except FileNotFoundError:
                pass
        return total


def _to_abs(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return ROOT_DIR / path


def _to_rel_text(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT_DIR.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _iter_image_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]


def _load_used_paths(embeddings_csv: Path) -> tuple[set[Path], set[Path], set[str]]:
    used_raw: set[Path] = set()
    used_crops: set[Path] = set()
    used_raw_rel: set[str] = set()

    with embeddings_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            is_valid = (row.get("is_valid_face") or "").strip().lower() in {"1", "true", "yes", "y"}
            if not is_valid:
                continue

            image_path = (row.get("image_path") or "").strip()
            if image_path:
                abs_path = _to_abs(image_path).resolve()
                used_raw.add(abs_path)
                used_raw_rel.add(_to_rel_text(abs_path))

            crop_path = (row.get("crop_path") or "").strip()
            if crop_path:
                used_crops.add(_to_abs(crop_path).resolve())

    return used_raw, used_crops, used_raw_rel


def build_plan(embeddings_csv: Path, raw_root: Path, crop_root: Path) -> PrunePlan:
    used_raw, used_crops, _ = _load_used_paths(embeddings_csv)
    raw_files = [path for path in _iter_image_files(raw_root) if path.resolve() not in used_raw]
    crop_files = [path for path in _iter_image_files(crop_root) if path.resolve() not in used_crops]
    return PrunePlan(raw_files=raw_files, crop_files=crop_files)


def _prune_empty_dirs(root: Path) -> int:
    if not root.exists():
        return 0
    removed = 0
    dirs = sorted((path for path in root.rglob("*") if path.is_dir()), key=lambda p: len(p.parts), reverse=True)
    for directory in dirs:
        try:
            directory.rmdir()
            removed += 1
        except OSError:
            pass
    return removed


def _update_manifest(manifest_path: Path, removed_raw: set[str]) -> int:
    if not manifest_path.exists() or not removed_raw:
        return 0

    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    if not fieldnames:
        return 0

    updated = 0
    for row in rows:
        row.pop(None, None)
        file_path = (row.get("file_path") or "").strip().replace("\\", "/")
        if file_path in removed_raw:
            row["file_path"] = ""
            row["status"] = "pruned_unused"
            updated += 1

    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    return updated


def apply_plan(plan: PrunePlan, raw_root: Path, crop_root: Path, manifest_path: Path | None) -> tuple[int, int, int]:
    removed_raw_rel = {_to_rel_text(path.resolve()) for path in plan.raw_files}

    raw_deleted = 0
    crop_deleted = 0
    for path in plan.raw_files:
        try:
            path.unlink()
            raw_deleted += 1
        except FileNotFoundError:
            pass
    for path in plan.crop_files:
        try:
            path.unlink()
            crop_deleted += 1
        except FileNotFoundError:
            pass

    _prune_empty_dirs(raw_root)
    _prune_empty_dirs(crop_root)
    manifest_updates = _update_manifest(manifest_path, removed_raw_rel) if manifest_path else 0
    return raw_deleted, crop_deleted, manifest_updates


def _format_bytes(value: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    size = float(value)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def main() -> None:
    parser = argparse.ArgumentParser(description="Delete raw/crop image files that are not used by valid embeddings.")
    parser.add_argument("--embeddings", default="data/image_embeddings.csv")
    parser.add_argument("--raw-root", default="data/raw_images")
    parser.add_argument("--crop-root", default="data/face_crops")
    parser.add_argument("--manifest", default="data/raw_images/manifest.csv")
    parser.add_argument("--apply", action="store_true", help="Actually delete files. Omit for dry-run.")
    args = parser.parse_args()

    embeddings_csv = _to_abs(args.embeddings)
    raw_root = _to_abs(args.raw_root)
    crop_root = _to_abs(args.crop_root)
    manifest_path = _to_abs(args.manifest) if args.manifest else None

    plan = build_plan(embeddings_csv=embeddings_csv, raw_root=raw_root, crop_root=crop_root)
    print(
        f"Unused raw files: {len(plan.raw_files)}; "
        f"unused crop files: {len(plan.crop_files)}; "
        f"reclaimable: {_format_bytes(plan.byte_count)}"
    )

    if not args.apply:
        print("Dry run only. Add --apply to delete these files.")
        return

    raw_deleted, crop_deleted, manifest_updates = apply_plan(
        plan=plan,
        raw_root=raw_root,
        crop_root=crop_root,
        manifest_path=manifest_path,
    )
    print(
        f"Deleted raw files: {raw_deleted}; "
        f"deleted crop files: {crop_deleted}; "
        f"manifest rows updated: {manifest_updates}"
    )


if __name__ == "__main__":
    main()
