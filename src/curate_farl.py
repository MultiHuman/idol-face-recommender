"""FaRL 임베딩 기반 이미지 필터링.

각 이미지의 FaRL 벡터와 해당 멤버의 FaRL 중심값(centroid) 사이의 코사인 유사도를 계산.
임계값 미만 이미지는 삭제(선택).

절차:
1) data/member_vectors_farl.csv 에서 각 멤버 centroid 로딩
2) data/image_farl.csv 순회하며 코사인 유사도 계산
3) 임계값 미만 이미지 삭제:
   - 원본 파일 삭제 (data/raw_images/{member_id}/xxx.jpg)
   - face crop 삭제 (data/face_crops/{member_id}/xxx.jpg)
   - data/image_embeddings.csv 에서 해당 행 is_valid_face=false 로 마킹
   - data/image_farl.csv 에서 해당 행 제거
   - data/manifest.csv 에서 file_path 공란 처리
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT_DIR = Path(__file__).resolve().parent.parent


@dataclass
class ImageRecord:
    member_id: str
    image_path: str
    vector: np.ndarray


def load_centroids(farl_vectors_csv: Path) -> dict[str, np.ndarray]:
    centroids: dict[str, np.ndarray] = {}
    with farl_vectors_csv.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            mid = (row.get("member_id") or "").strip()
            vec_json = (row.get("vector_json") or "").strip()
            if not mid or not vec_json:
                continue
            v = np.asarray(json.loads(vec_json), dtype=np.float32)
            centroids[mid] = v
    return centroids


def load_image_records(image_farl_csv: Path) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    with image_farl_csv.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            mid = (row.get("member_id") or "").strip()
            path = (row.get("image_path") or "").strip()
            vec_json = (row.get("farl_vector_json") or "").strip()
            if not mid or not path or not vec_json:
                continue
            vec = np.asarray(json.loads(vec_json), dtype=np.float32)
            records.append(ImageRecord(member_id=mid, image_path=path, vector=vec))
    return records


def mark_low_similarity(
    records: list[ImageRecord],
    centroids: dict[str, np.ndarray],
    min_similarity: float,
) -> list[tuple[ImageRecord, float]]:
    """임계값 미만 이미지 리턴: [(record, similarity), ...]"""
    marked: list[tuple[ImageRecord, float]] = []
    for rec in records:
        centroid = centroids.get(rec.member_id)
        if centroid is None:
            continue
        sim = float(np.dot(rec.vector, centroid))
        if sim < min_similarity:
            marked.append((rec, sim))
    return marked


def prune_empty_directories(start: Path, stop: Path) -> None:
    cur = start
    while cur != stop and cur.exists():
        try:
            if not any(cur.iterdir()):
                cur.rmdir()
            else:
                return
        except OSError:
            return
        cur = cur.parent


def delete_files(
    marked: list[tuple[ImageRecord, float]],
    raw_root: Path,
    crop_root: Path,
) -> int:
    deleted = 0
    for rec, _ in marked:
        raw_path = ROOT_DIR / rec.image_path
        if raw_path.exists():
            try:
                raw_path.unlink()
                deleted += 1
                prune_empty_directories(raw_path.parent, raw_root)
            except OSError:
                pass
        # face crop 파일명은 raw 파일명과 동일하지만 확장자가 .jpg 로 고정되는 경우가 있음
        crop_path = crop_root / rec.member_id / (raw_path.stem + raw_path.suffix)
        if crop_path.exists():
            try:
                crop_path.unlink()
                prune_empty_directories(crop_path.parent, crop_root)
            except OSError:
                pass
        else:
            # 확장자가 다를 수 있음 — 같은 stem 파일을 찾아서 삭제
            crop_dir = crop_root / rec.member_id
            if crop_dir.exists():
                for candidate in crop_dir.glob(f"{raw_path.stem}.*"):
                    try:
                        candidate.unlink()
                    except OSError:
                        pass
                prune_empty_directories(crop_dir, crop_root)
    return deleted


def update_embeddings_csv(embeddings_csv: Path, removed_paths: set[str]) -> int:
    if not embeddings_csv.exists() or not removed_paths:
        return 0
    with embeddings_csv.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    updated = 0
    for row in rows:
        row.pop(None, None)
        if (row.get("image_path") or "").strip() in removed_paths:
            row["is_valid_face"] = "false"
            row["vector_json"] = ""
            updated += 1
    with embeddings_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return updated


def update_farl_csv(farl_csv: Path, removed_paths: set[str]) -> int:
    if not farl_csv.exists() or not removed_paths:
        return 0
    with farl_csv.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    for r in rows:
        r.pop(None, None)
    kept = [r for r in rows if (r.get("image_path") or "").strip() not in removed_paths]
    removed = len(rows) - len(kept)
    with farl_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(kept)
    return removed


def update_manifest_csv(manifest_csv: Path, removed_paths: set[str]) -> int:
    if not manifest_csv.exists() or not removed_paths:
        return 0
    with manifest_csv.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    updated = 0
    for row in rows:
        row.pop(None, None)
        fp = (row.get("file_path") or "").strip()
        if fp and fp in removed_paths:
            row["file_path"] = ""
            row["status"] = "curated_farl"
            updated += 1
    with manifest_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return updated


def run(
    farl_vectors_csv: Path,
    image_farl_csv: Path,
    embeddings_csv: Path,
    manifest_csv: Path,
    raw_root: Path,
    crop_root: Path,
    min_similarity: float,
    dry_run: bool,
) -> None:
    centroids = load_centroids(farl_vectors_csv)
    print(f"Loaded {len(centroids)} member centroids.")

    records = load_image_records(image_farl_csv)
    print(f"Loaded {len(records)} image FaRL embeddings.")

    marked = mark_low_similarity(records, centroids, min_similarity)
    print(f"Marked {len(marked)} images for removal (FaRL cosine < {min_similarity}).")

    if dry_run:
        # 멤버별 상위 5개 출력
        from collections import defaultdict
        per_member: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for rec, sim in marked:
            per_member[rec.member_id].append((rec.image_path, sim))
        print("\nTop members with most removed images:")
        for mid, lst in sorted(per_member.items(), key=lambda x: -len(x[1]))[:15]:
            print(f"  {mid}: {len(lst)} removed (worst: {min(lst, key=lambda x: x[1])[1]:.3f})")
        print("\n(dry-run 모드. 파일 삭제하지 않음.)")
        return

    removed_paths = {rec.image_path for rec, _ in marked}
    deleted = delete_files(marked, raw_root, crop_root)
    print(f"Deleted {deleted} raw image files.")

    emb_updated = update_embeddings_csv(embeddings_csv, removed_paths)
    print(f"Updated {emb_updated} rows in {embeddings_csv.name} (is_valid_face=false).")

    farl_removed = update_farl_csv(image_farl_csv, removed_paths)
    print(f"Removed {farl_removed} rows from {image_farl_csv.name}.")

    manifest_updated = update_manifest_csv(manifest_csv, removed_paths)
    print(f"Updated {manifest_updated} rows in manifest.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Curate images by FaRL centroid similarity.")
    parser.add_argument("--farl-vectors", default="data/member_vectors_farl.csv")
    parser.add_argument("--image-farl", default="data/image_farl.csv")
    parser.add_argument("--embeddings", default="data/image_embeddings.csv")
    parser.add_argument("--manifest", default="data/raw_images/manifest.csv")
    parser.add_argument("--raw-root", default="data/raw_images")
    parser.add_argument("--crop-root", default="data/face_crops")
    parser.add_argument("--min-similarity", type=float, default=0.80)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(
        farl_vectors_csv=ROOT_DIR / args.farl_vectors,
        image_farl_csv=ROOT_DIR / args.image_farl,
        embeddings_csv=ROOT_DIR / args.embeddings,
        manifest_csv=ROOT_DIR / args.manifest,
        raw_root=ROOT_DIR / args.raw_root,
        crop_root=ROOT_DIR / args.crop_root,
        min_similarity=args.min_similarity,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
