"""image_farl.csv 의 per-image FaRL 임베딩을 멤버별로 집계.

개선점:
- spherical mean (Karcher) + IQR outlier trimming
- intra-member pairwise consistency (confidence) 저장
- kept_count, image_count 모두 기록
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.build_member_vectors import (
    _load_group_gender_map,
    _load_member_gender_map,
    apply_gender_overrides,
    resolve_group_gender_overrides,
    spherical_mean,
    trim_outliers,
)


def _parse_vector(vec_json: str) -> np.ndarray:
    vals = json.loads(vec_json)
    v = np.asarray(vals, dtype=np.float32)
    if v.ndim != 1 or v.size == 0:
        raise ValueError("Embedding vector must be a non-empty 1D array.")
    return v


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1e-12, None)


def _member_confidence(normalized: np.ndarray) -> float:
    n = normalized.shape[0]
    if n < 2:
        return 0.5
    sims = normalized @ normalized.T
    iu = np.triu_indices(n, k=1)
    return float(np.clip(sims[iu].mean(), 0.0, 1.0))


def build_farl_vectors(
    farl_csv: Path,
    embeddings_csv: Path,
    output_csv: Path,
    min_images: int = 5,
    trim_fraction: float = 0.15,
    min_keep_after_trim: int = 3,
    aggregator: str = "spherical",
    group_gender_csv: Path | None = None,
    member_gender_csv: Path | None = None,
    require_single_face: bool = True,
) -> int:
    if not farl_csv.exists():
        raise FileNotFoundError(f"FaRL CSV not found: {farl_csv}")
    if not embeddings_csv.exists():
        raise FileNotFoundError(f"Embeddings CSV not found: {embeddings_csv}")

    meta: dict[tuple[str, str], dict[str, str]] = {}
    with embeddings_csv.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            mid = (row.get("member_id") or "").strip()
            img = (row.get("image_path") or "").strip()
            if not mid or not img:
                continue
            # is_valid_face=false 는 앵커 필터 등으로 제거된 이미지. 절대 포함 X.
            is_valid = (row.get("is_valid_face") or "").strip().lower() in {"1", "true", "yes", "y"}
            if not is_valid:
                continue
            if require_single_face:
                try:
                    face_count = int((row.get("face_count") or "1").strip() or "1")
                except ValueError:
                    face_count = 1
                if face_count != 1:
                    continue
            meta[(mid, img)] = {
                "group_name": (row.get("group_name") or "").strip(),
                "member_name": (row.get("member_name") or "").strip(),
                "quality_score": row.get("quality_score") or "0.0",
                "gender": (row.get("gender") or "").strip(),
            }

    grouped: dict[str, dict[str, object]] = defaultdict(
        lambda: {"group_name": "", "member_name": "", "genders": [], "vectors": []}
    )
    with farl_csv.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            mid = (row.get("member_id") or "").strip()
            img = (row.get("image_path") or "").strip()
            vec_json = (row.get("farl_vector_json") or "").strip()
            if not mid or not vec_json:
                continue

            m = meta.get((mid, img))
            if m is None:
                continue
            try:
                quality = float(m["quality_score"])
            except ValueError:
                quality = 0.0

            try:
                vec = _parse_vector(vec_json)
            except ValueError:
                continue

            entry = grouped[mid]
            entry["group_name"] = m["group_name"]
            entry["member_name"] = m["member_name"]
            if m.get("gender"):
                entry["genders"].append(m["gender"])
            entry["vectors"].append((vec, quality))

    rows_to_write: list[dict[str, object]] = []
    for mid, entry in grouped.items():
        vecs: list[tuple[np.ndarray, float]] = entry["vectors"]  # type: ignore[assignment]
        if len(vecs) < min_images:
            continue
        matrix = np.vstack([v for v, _ in vecs])
        normalized = _normalize_rows(matrix)
        weights = np.array([max(w, 1e-6) for _, w in vecs], dtype=np.float32)

        normalized_trim, weights_trim = trim_outliers(
            normalized, weights, trim_fraction=trim_fraction, min_keep=min_keep_after_trim
        )

        if aggregator == "mean":
            w = weights_trim / weights_trim.sum()
            avg = (normalized_trim * w[:, None]).sum(axis=0)
            avg = avg / max(np.linalg.norm(avg), 1e-12)
        else:
            avg = spherical_mean(normalized_trim)

        confidence = _member_confidence(normalized_trim)

        genders = entry["genders"]  # type: ignore[assignment]
        if genders:
            from collections import Counter
            dominant_gender = Counter(genders).most_common(1)[0][0]
        else:
            dominant_gender = ""

        rows_to_write.append({
            "member_id": mid,
            "group_name": str(entry["group_name"]),
            "member_name": str(entry["member_name"]),
            "image_count": len(vecs),
            "kept_count": int(normalized_trim.shape[0]),
            "confidence": f"{confidence:.4f}",
            "gender": dominant_gender,
            "vector_json": json.dumps(avg.tolist(), separators=(",", ":")),
        })

    # 그룹 성별 확정: external ground truth 우선, 없으면 다수결
    external_gender = _load_group_gender_map(group_gender_csv) if group_gender_csv else {}
    group_gender = resolve_group_gender_overrides(
        rows_to_write=[dict(row) for row in rows_to_write],
        external_gender=external_gender,
    )
    member_gender = _load_member_gender_map(member_gender_csv) if member_gender_csv else {}
    apply_gender_overrides(
        rows_to_write=[row for row in rows_to_write],
        group_gender=group_gender,
        member_gender=member_gender,
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["member_id", "group_name", "member_name", "image_count", "kept_count", "confidence", "gender", "vector_json"]
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows_to_write)

    return len(rows_to_write)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate per-image FaRL embeddings into member-level vectors.")
    parser.add_argument("--farl", default="data/image_farl.csv")
    parser.add_argument("--embeddings", default="data/image_embeddings.csv")
    parser.add_argument("--output", default="data/member_vectors_farl.csv")
    parser.add_argument("--min-images", type=int, default=5)
    parser.add_argument("--trim-fraction", type=float, default=0.15)
    parser.add_argument("--min-keep-after-trim", type=int, default=3)
    parser.add_argument("--aggregator", choices=["spherical", "mean"], default="spherical")
    parser.add_argument("--group-gender", default="data/group_genders.csv")
    parser.add_argument("--member-gender", default="data/member_genders.csv")
    parser.add_argument(
        "--allow-multi-face",
        action="store_true",
        help="face_count > 1 row도 FaRL 멤버 벡터 집계에 포함한다.",
    )
    args = parser.parse_args()

    count = build_farl_vectors(
        farl_csv=Path(args.farl),
        embeddings_csv=Path(args.embeddings),
        output_csv=Path(args.output),
        min_images=args.min_images,
        trim_fraction=args.trim_fraction,
        min_keep_after_trim=args.min_keep_after_trim,
        aggregator=args.aggregator,
        group_gender_csv=Path(args.group_gender) if args.group_gender else None,
        member_gender_csv=Path(args.member_gender) if args.member_gender else None,
        require_single_face=not args.allow_multi_face,
    )
    print(f"Wrote {count} member FaRL vectors to {args.output}")


if __name__ == "__main__":
    main()
