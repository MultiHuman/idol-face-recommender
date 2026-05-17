"""Per-image ArcFace 임베딩을 member vector로 집계.

개선점 (vs 단순 평균):
1) L2 정규화 후 spherical (Karcher) mean 으로 집계. 단위구 위에서 일관성 있는 중심.
2) IQR 기반 outlier trimming — centroid 에서 떨어진 상위 25% 는 평균 계산에서 제외.
3) intra-member pairwise cosine (consistency) 와 image_count 로 신뢰도 점수 산출.
4) 최종 vector 를 L2 normalize 해서 코사인 공간에 둔다.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y"}


def _parse_vector(vector_json: str) -> np.ndarray:
    values = json.loads(vector_json)
    vector = np.asarray(values, dtype=np.float32)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError("Embedding vector must be a non-empty 1D array.")
    return vector


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1e-12, None)


def spherical_mean(vectors: np.ndarray, iters: int = 8) -> np.ndarray:
    """단위구 위 Karcher mean. vectors 는 미리 L2 정규화돼야 한다."""
    if vectors.shape[0] == 1:
        return vectors[0].copy()
    m = vectors.mean(axis=0)
    m /= max(np.linalg.norm(m), 1e-12)
    for _ in range(iters):
        dots = np.clip(vectors @ m, -1.0, 1.0)
        # tangent space projection
        tangents = vectors - dots[:, None] * m
        update = tangents.mean(axis=0)
        m = m + update
        m /= max(np.linalg.norm(m), 1e-12)
    return m


def trim_outliers(
    normalized: np.ndarray,
    weights: np.ndarray,
    trim_fraction: float,
    min_keep: int,
) -> tuple[np.ndarray, np.ndarray]:
    """centroid cosine 하위 `trim_fraction` 제거. 남는 개수가 min_keep 미만이면 잘라내지 않는다."""
    n = normalized.shape[0]
    if n <= min_keep or trim_fraction <= 0:
        return normalized, weights
    centroid = spherical_mean(normalized)
    sims = normalized @ centroid
    keep_n = max(min_keep, int(round(n * (1.0 - trim_fraction))))
    order = np.argsort(-sims)  # 유사도 높은 순
    keep_idx = np.sort(order[:keep_n])
    return normalized[keep_idx], weights[keep_idx]


def _member_confidence(normalized: np.ndarray) -> float:
    """intra-member pairwise cosine 평균. 1장이면 0.5 (중립), 2장 이상이면 실제 평균."""
    n = normalized.shape[0]
    if n < 2:
        return 0.5
    sims = normalized @ normalized.T
    iu = np.triu_indices(n, k=1)
    return float(np.clip(sims[iu].mean(), 0.0, 1.0))


def _load_group_gender_map(path: str | Path) -> dict[str, str]:
    path = Path(path)
    if not path.exists():
        return {}
    result: dict[str, str] = {}
    with path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            g = (row.get("group_name") or "").strip()
            gender = (row.get("gender") or "").strip().upper()
            if g and gender in ("F", "M", "COED"):
                result[g] = gender
    return result


def _load_member_gender_map(path: str | Path) -> dict[str, str]:
    path = Path(path)
    if not path.exists():
        return {}
    result: dict[str, str] = {}
    with path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            member_id = (row.get("member_id") or "").strip()
            gender = (row.get("gender") or "").strip().upper()
            if member_id and gender in ("F", "M"):
                result[member_id] = gender
    return result


def resolve_group_gender_overrides(
    rows_to_write: list[dict[str, object]],
    external_gender: dict[str, str],
) -> dict[str, str]:
    coed_groups = {group for group, gender in external_gender.items() if gender == "COED"}
    external_binary_gender = {
        group: gender for group, gender in external_gender.items() if gender in ("F", "M")
    }

    from collections import Counter

    group_votes: dict[str, Counter] = defaultdict(Counter)
    for row in rows_to_write:
        g = row["group_name"]
        if g and row.get("gender"):
            group_votes[str(g)][row["gender"]] += 1

    group_gender: dict[str, str] = {}
    for g, counter in group_votes.items():
        if g in coed_groups:
            continue
        total = sum(counter.values())
        top_gender, top_count = counter.most_common(1)[0]
        if total > 0 and top_count / total >= 0.5:
            group_gender[g] = top_gender

    group_gender.update(external_binary_gender)
    return group_gender


def apply_gender_overrides(
    rows_to_write: list[dict[str, object]],
    group_gender: dict[str, str],
    member_gender: dict[str, str],
) -> None:
    for row in rows_to_write:
        group_name = str(row.get("group_name") or "")
        if group_name in group_gender:
            row["gender"] = group_gender[group_name]

        member_id = str(row.get("member_id") or "")
        if member_id in member_gender:
            row["gender"] = member_gender[member_id]


def build_member_vectors(
    input_csv: str | Path,
    output_csv: str | Path,
    min_images: int = 5,
    min_quality: float = 0.0,
    trim_fraction: float = 0.0,
    min_keep_after_trim: int = 3,
    aggregator: str = "spherical",
    group_gender_csv: str | Path | None = None,
    member_gender_csv: str | Path | None = None,
    require_single_face: bool = True,
) -> int:
    source = Path(input_csv)
    destination = Path(output_csv)
    if not source.exists():
        raise FileNotFoundError(f"Input file does not exist: {source}")

    grouped: dict[str, dict[str, object]] = defaultdict(
        lambda: {"group_name": "", "member_name": "", "genders": [], "vectors": []}
    )

    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if not _parse_bool(row.get("is_valid_face") or "false"):
                continue

            if require_single_face:
                try:
                    face_count = int((row.get("face_count") or "1").strip() or "1")
                except ValueError:
                    face_count = 1
                if face_count != 1:
                    continue

            quality_score = float(row.get("quality_score") or 0.0)
            if quality_score < min_quality:
                continue

            member_id = (row.get("member_id") or "").strip()
            if not member_id:
                continue

            entry = grouped[member_id]
            entry["group_name"] = (row.get("group_name") or "").strip()
            entry["member_name"] = (row.get("member_name") or "").strip()
            gender = (row.get("gender") or "").strip()
            if gender:
                entry["genders"].append(gender)
            entry["vectors"].append((_parse_vector(row.get("vector_json") or "[]"), quality_score))

    rows_to_write: list[dict[str, str | int]] = []
    for member_id, entry in grouped.items():
        vectors: list[tuple[np.ndarray, float]] = entry["vectors"]  # type: ignore[assignment]
        if len(vectors) < min_images:
            continue

        matrix = np.vstack([v for v, _ in vectors])
        normalized = _normalize_rows(matrix)
        weights = np.array([max(w, 1e-6) for _, w in vectors], dtype=np.float32)

        normalized_trim, weights_trim = trim_outliers(
            normalized, weights, trim_fraction=trim_fraction, min_keep=min_keep_after_trim
        )

        if aggregator == "mean":
            w = weights_trim / weights_trim.sum()
            avg = (normalized_trim * w[:, None]).sum(axis=0)
            avg = avg / max(np.linalg.norm(avg), 1e-12)
        else:  # spherical
            avg = spherical_mean(normalized_trim)

        confidence = _member_confidence(normalized_trim)

        genders = entry["genders"]  # type: ignore[assignment]
        # 멤버의 성별은 다수결 (비어있으면 빈 문자열)
        if genders:
            from collections import Counter
            dominant_gender = Counter(genders).most_common(1)[0][0]
        else:
            dominant_gender = ""

        rows_to_write.append(
            {
                "member_id": member_id,
                "group_name": str(entry["group_name"]),
                "member_name": str(entry["member_name"]),
                "image_count": len(vectors),
                "kept_count": int(normalized_trim.shape[0]),
                "confidence": f"{confidence:.4f}",
                "gender": dominant_gender,
                "vector_json": json.dumps(avg.tolist(), ensure_ascii=True, separators=(",", ":")),
            }
        )

    # 그룹 성별 확정 절차 (우선순위):
    # 1) group_gender_csv 가 있으면 외부 ground truth 로 덮어씀 (kprofiles "boy/girl group" 텍스트 기반)
    # 2) 그게 없으면 그룹 멤버들의 genderage 다수결로 덮어씀
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

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["member_id", "group_name", "member_name", "image_count", "kept_count", "confidence", "gender", "vector_json"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows_to_write)

    return len(rows_to_write)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate image embeddings into member vectors.")
    parser.add_argument("--input", default="data/image_embeddings.csv")
    parser.add_argument("--output", default="data/member_vectors.csv")
    parser.add_argument("--min-images", type=int, default=5)
    parser.add_argument("--min-quality", type=float, default=0.0)
    parser.add_argument(
        "--trim-fraction",
        type=float,
        default=0.15,
        help="centroid 유사도 하위 fraction 을 평균 계산에서 제외 (0 이면 비활성).",
    )
    parser.add_argument(
        "--min-keep-after-trim",
        type=int,
        default=3,
        help="trim 후 남을 최소 이미지 수.",
    )
    parser.add_argument(
        "--aggregator",
        choices=["spherical", "mean"],
        default="spherical",
        help="spherical = Karcher mean (권장), mean = quality-weighted L2-normalized mean.",
    )
    parser.add_argument(
        "--group-gender",
        default="data/group_genders.csv",
        help="(선택) group_name → gender ground truth CSV. genderage 다수결을 덮어씀.",
    )
    parser.add_argument(
        "--member-gender",
        default="data/member_genders.csv",
        help="(선택) member_id → gender ground truth CSV. group override보다 우선.",
    )
    parser.add_argument(
        "--allow-multi-face",
        action="store_true",
        help="face_count > 1 row도 멤버 벡터 집계에 포함한다.",
    )
    args = parser.parse_args()

    count = build_member_vectors(
        input_csv=args.input,
        output_csv=args.output,
        min_images=args.min_images,
        min_quality=args.min_quality,
        trim_fraction=args.trim_fraction,
        min_keep_after_trim=args.min_keep_after_trim,
        aggregator=args.aggregator,
        group_gender_csv=args.group_gender,
        member_gender_csv=args.member_gender,
        require_single_face=not args.allow_multi_face,
    )
    print(f"Wrote {count} member vectors to {args.output}")


if __name__ == "__main__":
    main()
