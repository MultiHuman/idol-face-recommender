from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT_DIR = Path(__file__).resolve().parent.parent


@dataclass
class EmbeddingRecord:
    member_id: str
    group_name: str
    member_name: str
    image_path_text: str
    crop_path_text: str
    image_path: Path
    crop_path: Path | None
    is_valid_face: bool
    face_count: int
    quality_score: float
    vector: np.ndarray | None
    centroid_similarity: float | None = None
    removal_reason: str = ""


def parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y"}


def to_abs_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return ROOT_DIR / path


def parse_vector(vector_json: str) -> np.ndarray | None:
    vector_json = vector_json.strip()
    if not vector_json:
        return None
    values = json.loads(vector_json)
    vector = np.asarray(values, dtype=np.float32)
    if vector.ndim != 1 or vector.size == 0:
        return None
    return vector


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1e-12, None)


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError("Vector norm is zero.")
    return vector / norm


def prune_empty_directories(start_dir: Path, stop_dir: Path) -> None:
    current = start_dir
    stop_resolved = stop_dir.resolve()
    while True:
        try:
            current_resolved = current.resolve()
        except FileNotFoundError:
            break
        if current_resolved == stop_resolved:
            break
        try:
            current.rmdir()
        except OSError:
            break
        current = current.parent


def load_records(csv_path: str | Path) -> list[EmbeddingRecord]:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Embeddings CSV does not exist: {path}")

    records: list[EmbeddingRecord] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            image_path_text = (row.get("image_path") or "").strip()
            if not image_path_text:
                continue

            crop_path_text = (row.get("crop_path") or "").strip()
            records.append(
                EmbeddingRecord(
                    member_id=(row.get("member_id") or "").strip(),
                    group_name=(row.get("group_name") or "").strip(),
                    member_name=(row.get("member_name") or "").strip(),
                    image_path_text=image_path_text,
                    crop_path_text=crop_path_text,
                    image_path=to_abs_path(image_path_text),
                    crop_path=to_abs_path(crop_path_text) if crop_path_text else None,
                    is_valid_face=parse_bool(row.get("is_valid_face") or "false"),
                    face_count=int((row.get("face_count") or "0").strip() or "0"),
                    quality_score=float((row.get("quality_score") or "0").strip() or "0"),
                    vector=parse_vector(row.get("vector_json") or ""),
                )
            )
    return records


def compute_centroid_similarities(records: list[EmbeddingRecord]) -> None:
    grouped: dict[str, list[EmbeddingRecord]] = {}
    for record in records:
        if record.removal_reason or not record.is_valid_face or record.vector is None:
            continue
        grouped.setdefault(record.member_id, []).append(record)

    for member_records in grouped.values():
        if not member_records:
            continue
        matrix = np.vstack([record.vector for record in member_records if record.vector is not None])
        normalized = normalize_rows(matrix)
        centroid = normalize_vector(matrix.mean(axis=0))
        similarities = normalized @ centroid
        for record, similarity in zip(member_records, similarities.tolist(), strict=True):
            record.centroid_similarity = float(similarity)


def mark_low_quality_records(
    records: list[EmbeddingRecord],
    remove_invalid: bool,
    min_quality: float,
    min_centroid_similarity: float | None,
    min_keep_per_member: int,
) -> None:
    for record in records:
        if remove_invalid and not record.is_valid_face:
            record.removal_reason = "invalid_face_result"
            continue
        if not record.is_valid_face or record.vector is None:
            continue
        if record.quality_score < min_quality:
            record.removal_reason = f"quality_score<{min_quality:.2f}"

    if min_centroid_similarity is None:
        return

    compute_centroid_similarities(records)
    grouped: dict[str, list[EmbeddingRecord]] = {}
    for record in records:
        if record.removal_reason or not record.is_valid_face or record.vector is None:
            continue
        grouped.setdefault(record.member_id, []).append(record)

    for member_records in grouped.values():
        if len(member_records) <= min_keep_per_member:
            continue

        flagged = [
            record
            for record in member_records
            if record.centroid_similarity is not None and record.centroid_similarity < min_centroid_similarity
        ]
        if not flagged:
            continue

        removable_slots = max(0, len(member_records) - min_keep_per_member)
        if removable_slots == 0:
            continue

        flagged.sort(
            key=lambda record: (
                float(record.centroid_similarity if record.centroid_similarity is not None else -1.0),
                record.quality_score,
            )
        )
        for record in flagged[:removable_slots]:
            record.removal_reason = f"centroid_similarity<{min_centroid_similarity:.2f}"


def _quarantine_or_delete(path: Path, quarantine_root: Path | None) -> bool:
    if not path.exists():
        return False
    if quarantine_root is None:
        path.unlink()
        return True

    rel = path.resolve().relative_to(ROOT_DIR.resolve())
    destination = quarantine_root / rel
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination = destination.with_name(f"{destination.stem}_{os.getpid()}{destination.suffix}")
    shutil.move(str(path), str(destination))
    return True


def delete_marked_files(
    records: list[EmbeddingRecord],
    raw_root: Path,
    crop_root: Path,
    manifest_path: Path,
    log_output: Path,
    quarantine_root: Path | None = None,
) -> int:
    rows_to_log: list[dict[str, str]] = []
    deleted_count = 0

    for record in records:
        if not record.removal_reason:
            continue

        raw_deleted = False
        crop_deleted = False

        try:
            raw_deleted = _quarantine_or_delete(record.image_path, quarantine_root)
            if raw_deleted:
                deleted_count += 1
        except (FileNotFoundError, ValueError):
            raw_deleted = False

        if record.crop_path is not None:
            try:
                crop_deleted = _quarantine_or_delete(record.crop_path, quarantine_root)
            except (FileNotFoundError, ValueError):
                crop_deleted = False

        if raw_deleted:
            prune_empty_directories(record.image_path.parent, raw_root)
        if crop_deleted and record.crop_path is not None:
            prune_empty_directories(record.crop_path.parent, crop_root)

        rows_to_log.append(
            {
                "member_id": record.member_id,
                "group_name": record.group_name,
                "member_name": record.member_name,
                "image_path": record.image_path_text,
                "crop_path": record.crop_path_text,
                "is_valid_face": str(record.is_valid_face).lower(),
                "face_count": str(record.face_count),
                "quality_score": f"{record.quality_score:.4f}",
                "centroid_similarity": (
                    f"{record.centroid_similarity:.4f}" if record.centroid_similarity is not None else ""
                ),
                "removal_reason": record.removal_reason,
                "raw_deleted": str(raw_deleted).lower(),
                "crop_deleted": str(crop_deleted).lower(),
            }
        )

    log_output.parent.mkdir(parents=True, exist_ok=True)
    with log_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "member_id",
                "group_name",
                "member_name",
                "image_path",
                "crop_path",
                "is_valid_face",
                "face_count",
                "quality_score",
                "centroid_similarity",
                "removal_reason",
                "raw_deleted",
                "crop_deleted",
            ],
        )
        writer.writeheader()
        writer.writerows(rows_to_log)

    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = list(reader.fieldnames or [])
            rows = list(reader)

        # URL은 유지하되 file_path만 지워서 재다운로드 방지
        removed_paths = {record.image_path_text for record in records if record.removal_reason}
        kept_rows = []
        for row in rows:
            row.pop(None, None)
            file_path = (row.get("file_path") or "").strip()
            if file_path and file_path in removed_paths:
                row = dict(row)
                row["file_path"] = ""
                row["status"] = "curated"
            kept_rows.append(row)

        if fieldnames:
            with manifest_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(kept_rows)

    return deleted_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Delete invalid or low-quality idol images from the dataset.")
    parser.add_argument(
        "--embeddings",
        default="data/image_embeddings.csv",
        help="Path to the image-level embeddings CSV file.",
    )
    parser.add_argument(
        "--raw-root",
        default="data/raw_images",
        help="Directory that stores raw images.",
    )
    parser.add_argument(
        "--crop-root",
        default="data/face_crops",
        help="Directory that stores face crops.",
    )
    parser.add_argument(
        "--manifest",
        default="data/raw_images/manifest.csv",
        help="Manifest CSV file to keep in sync with the remaining raw images.",
    )
    parser.add_argument(
        "--log-output",
        default="data/curation_log.csv",
        help="CSV file that records which images were removed.",
    )
    parser.add_argument(
        "--min-quality",
        type=float,
        default=0.0,
        help="Delete valid images whose quality score is below this value.",
    )
    parser.add_argument(
        "--min-centroid-similarity",
        type=float,
        default=0.0,
        help="Delete valid images whose member-centroid cosine similarity is below this value.",
    )
    parser.add_argument(
        "--min-keep-per-member",
        type=int,
        default=3,
        help="Minimum number of valid images to keep per member when centroid filtering is enabled.",
    )
    parser.add_argument(
        "--keep-invalid",
        action="store_true",
        help="Do not delete rows marked as invalid face results.",
    )
    parser.add_argument("--member-ids", nargs="*", help="Only curate these member IDs.")
    parser.add_argument("--quarantine-dir", default=None, help="Move removed files here instead of deleting them.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be removed without deleting files.",
    )
    args = parser.parse_args()

    records = load_records(args.embeddings)
    if args.member_ids:
        selected = {mid.strip() for mid in args.member_ids if mid.strip()}
        records = [record for record in records if record.member_id in selected]
    min_centroid_similarity = args.min_centroid_similarity if args.min_centroid_similarity > 0 else None
    mark_low_quality_records(
        records=records,
        remove_invalid=not args.keep_invalid,
        min_quality=args.min_quality,
        min_centroid_similarity=min_centroid_similarity,
        min_keep_per_member=args.min_keep_per_member,
    )

    marked = [record for record in records if record.removal_reason]
    print(f"Marked {len(marked)} images for removal.")
    reason_counts: dict[str, int] = {}
    for record in marked:
        reason_counts[record.removal_reason] = reason_counts.get(record.removal_reason, 0) + 1
    for reason, count in sorted(reason_counts.items()):
        print(f"- {reason}: {count}")

    if args.dry_run:
        print("Dry run only. No files were deleted.")
        return

    deleted_count = delete_marked_files(
        records=records,
        raw_root=to_abs_path(args.raw_root),
        crop_root=to_abs_path(args.crop_root),
        manifest_path=to_abs_path(args.manifest),
        log_output=to_abs_path(args.log_output),
        quarantine_root=to_abs_path(args.quarantine_dir) if args.quarantine_dir else None,
    )
    action = "Quarantined" if args.quarantine_dir else "Deleted"
    print(f"{action} {deleted_count} raw images. Log: {args.log_output}")

    # 삭제된 항목을 image_embeddings.csv에서 is_valid_face=false로 업데이트
    removed_paths = {r.image_path_text for r in records if r.removal_reason}
    if removed_paths:
        emb_path = to_abs_path(args.embeddings)
        with emb_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = list(reader.fieldnames or [])
            rows = list(reader)
        updated = 0
        for row in rows:
            # DictReader restkey 로 들어간 extras 제거 (CSV 헤더와 row 폭 불일치 방어)
            row.pop(None, None)
            if (row.get("image_path") or "").strip() in removed_paths:
                row["is_valid_face"] = "false"
                row["vector_json"] = ""
                updated += 1
        if fieldnames:
            with emb_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(rows)
        print(f"Updated {updated} rows in embeddings CSV.")


if __name__ == "__main__":
    main()
