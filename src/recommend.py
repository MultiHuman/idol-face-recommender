"""멤버 추천 엔진.

개선점 (vs 단순 코사인 평균):
1) taste_vector 를 spherical mean (Karcher) 로 계산 — 단위구 위 올바른 평균.
2) ArcFace + FaRL (+ 선택적 landmark) 를 z-score 정규화 후 가중 합산.
3) 멤버 confidence (intra-pairwise cosine) 로 최종 점수 가중.
4) MMR 재순위 + 그룹당 최대 N 제약으로 추천 다양성 확보.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class MemberVector:
    member_id: str
    group_name: str
    member_name: str
    image_count: int
    confidence: float
    gender: str
    vector: np.ndarray


def _parse_vector(vector_json: str) -> np.ndarray:
    values = json.loads(vector_json)
    vector = np.asarray(values, dtype=np.float32)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError("Embedding vector must be a non-empty 1D array.")
    return vector


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError("Zero vector cannot be normalized.")
    return vector / norm


def _spherical_mean(vectors: np.ndarray, iters: int = 8) -> np.ndarray:
    """Karcher mean on unit sphere."""
    if vectors.shape[0] == 1:
        return vectors[0].copy()
    m = vectors.mean(axis=0)
    m /= max(np.linalg.norm(m), 1e-12)
    for _ in range(iters):
        dots = np.clip(vectors @ m, -1.0, 1.0)
        tangents = vectors - dots[:, None] * m
        m = m + tangents.mean(axis=0)
        m /= max(np.linalg.norm(m), 1e-12)
    return m


def load_member_vectors(csv_path: str | Path) -> list[MemberVector]:
    path = Path(csv_path)
    if not path.exists():
        return []

    members: list[MemberVector] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            member_id = (row.get("member_id") or "").strip()
            if not member_id:
                continue

            vector_json = (row.get("vector_json") or "").strip()
            if not vector_json:
                continue

            try:
                conf = float((row.get("confidence") or "0.5").strip())
            except ValueError:
                conf = 0.5

            members.append(
                MemberVector(
                    member_id=member_id,
                    group_name=(row.get("group_name") or "").strip(),
                    member_name=(row.get("member_name") or "").strip(),
                    image_count=int(row.get("image_count") or 0),
                    confidence=conf,
                    gender=(row.get("gender") or "").strip(),
                    vector=_parse_vector(vector_json),
                )
            )

    return members


def _ensure_same_dimension(members: Iterable[MemberVector]) -> None:
    sizes = {member.vector.size for member in members}
    if len(sizes) > 1:
        raise ValueError("All member vectors must have the same dimension.")


def build_taste_vector(members: list[MemberVector], liked_member_ids: Iterable[str]) -> np.ndarray:
    """좋아하는 멤버들의 spherical mean taste vector."""
    liked_ids = {member_id.strip() for member_id in liked_member_ids if member_id.strip()}
    liked = [member for member in members if member.member_id in liked_ids]
    if not liked:
        raise ValueError("No matching members were found for the selected favorite IDs.")

    normalized = np.vstack([_normalize(m.vector) for m in liked])
    return _spherical_mean(normalized)


def _zscore(values: np.ndarray, background_mean: float, background_std: float) -> np.ndarray:
    if background_std < 1e-6:
        return values - background_mean
    return (values - background_mean) / background_std


def _background_stats(vectors: np.ndarray, taste: np.ndarray, sample_size: int = 500) -> tuple[float, float]:
    """taste vs 전체 멤버 코사인 분포의 mean/std (모델별 절대값 다르니 정규화용)."""
    n = vectors.shape[0]
    if n <= sample_size:
        sims = vectors @ taste
    else:
        rng = np.random.default_rng(seed=42)
        idx = rng.choice(n, sample_size, replace=False)
        sims = vectors[idx] @ taste
    return float(sims.mean()), float(sims.std())


def compute_scores(
    primary_members: list[MemberVector],
    liked_member_ids: list[str],
    secondary_members: list[MemberVector] | None = None,
    tertiary_members: list[MemberVector] | None = None,
    weights: tuple[float, ...] = (1.0,),
    use_confidence_weight: bool = True,
) -> dict[str, dict[str, float]]:
    """여러 임베딩 세트의 z-score 합산. primary 는 필수, secondary/tertiary 는 옵션.

    리턴: {member_id: {"score_<i>": z, "score": final_z_sum, "confidence": c}}
    """
    if not primary_members:
        return {}

    sources = [primary_members]
    for s in (secondary_members, tertiary_members):
        if s:
            sources.append(s)

    liked_set = set(liked_member_ids)

    # 각 source 별로 z-score 계산
    per_source_scores: list[dict[str, float]] = []
    per_source_weights: list[float] = []
    for si, src in enumerate(sources):
        _ensure_same_dimension(src)
        liked_vectors = [_normalize(m.vector) for m in src if m.member_id in liked_set]
        if not liked_vectors:
            # 이 source 에는 liked 멤버가 하나도 없으면 skip
            continue
        mat = np.vstack([_normalize(m.vector) for m in src])
        taste = _spherical_mean(np.vstack(liked_vectors))
        sims = mat @ taste
        bg_mean, bg_std = _background_stats(mat, taste)
        zs = _zscore(sims, bg_mean, bg_std)
        per_source_scores.append({m.member_id: float(zs[i]) for i, m in enumerate(src)})
        per_source_weights.append(weights[si] if si < len(weights) else 1.0)

    if not per_source_scores:
        return {}

    # 합산
    total_weight = sum(abs(w) for w in per_source_weights) or 1.0
    out: dict[str, dict[str, float]] = {}
    for m in primary_members:
        mid = m.member_id
        total = 0.0
        per: dict[str, float] = {}
        for si, scores in enumerate(per_source_scores):
            v = scores.get(mid)
            if v is None:
                continue
            w = per_source_weights[si]
            per[f"score_{si}"] = v
            total += w * v
        fused = total / total_weight

        if use_confidence_weight:
            # low-confidence 멤버는 점수 스케일 축소, 절대값이 커지는 걸 막는다
            fused = fused * (0.5 + 0.5 * m.confidence)

        per["score"] = fused
        per["confidence"] = m.confidence
        out[mid] = per
    return out


def _fused_embedding(
    mid: str,
    members_by_id: dict[str, MemberVector],
    extra_by_id: list[dict[str, MemberVector]],
) -> np.ndarray | None:
    """primary + secondary/tertiary 정규화 벡터를 concat → 다시 정규화.
    Fusion 공간에서 redundancy 계산에 사용."""
    parts: list[np.ndarray] = []
    mv = members_by_id.get(mid)
    if mv is None:
        return None
    parts.append(_normalize(mv.vector))
    for extra in extra_by_id:
        em = extra.get(mid)
        if em is not None:
            parts.append(_normalize(em.vector))
    combined = np.concatenate(parts)
    return _normalize(combined)


def mmr_rerank(
    candidates: list[dict[str, float | str | int]],
    members_by_id: dict[str, MemberVector],
    lambda_relevance: float,
    top_k: int,
    max_per_group: int,
    extra_by_id: list[dict[str, MemberVector]] | None = None,
) -> list[dict[str, float | str | int]]:
    """Max Marginal Relevance + 그룹당 최대 N 제약.

    extra_by_id 가 주어지면 primary + secondary/tertiary 벡터를 concat 해서
    fusion 공간에서 redundancy 를 계산. 없으면 primary 만."""
    if not candidates:
        return []

    extra_by_id = extra_by_id or []

    selected: list[dict[str, float | str | int]] = []
    remaining = list(candidates)
    group_counts: dict[str, int] = {}

    selected_vectors: list[np.ndarray] = []
    while remaining and len(selected) < top_k:
        best_idx = -1
        best_score = -1e9
        for idx, cand in enumerate(remaining):
            mid = str(cand["member_id"])
            mv = members_by_id.get(mid)
            if mv is None:
                continue
            grp = mv.group_name
            if max_per_group > 0 and group_counts.get(grp, 0) >= max_per_group:
                continue

            rel = float(cand["score"])
            if selected_vectors:
                fused = _fused_embedding(mid, members_by_id, extra_by_id)
                if fused is None:
                    fused = _normalize(mv.vector)
                sims = [float(fused @ sv) for sv in selected_vectors]
                redundancy = max(sims)
            else:
                redundancy = 0.0

            mmr = lambda_relevance * rel - (1.0 - lambda_relevance) * redundancy
            if mmr > best_score:
                best_score = mmr
                best_idx = idx

        if best_idx < 0:
            break
        chosen = remaining.pop(best_idx)
        mid = str(chosen["member_id"])
        mv = members_by_id[mid]
        selected.append(chosen)
        fused = _fused_embedding(mid, members_by_id, extra_by_id)
        selected_vectors.append(fused if fused is not None else _normalize(mv.vector))
        group_counts[mv.group_name] = group_counts.get(mv.group_name, 0) + 1
    return selected


def recommend_from_members(
    members: list[MemberVector],
    liked_member_ids: Iterable[str],
    top_k: int = 10,
    secondary_members: list[MemberVector] | None = None,
    tertiary_members: list[MemberVector] | None = None,
    weights: tuple[float, ...] = (1.0,),
    use_confidence_weight: bool = True,
    mmr_lambda: float = 1.0,
    max_per_group: int = 0,
    pool_size: int = 50,
    gender_filter: str = "auto",
    min_image_count: int = 0,
    min_confidence: float = 0.0,
) -> list[dict[str, str | int | float]]:
    """gender_filter: 'auto' = liked 멤버들의 다수 성별과 일치하는 멤버만, 'off' = 필터 없음, 'F'/'M' = 강제 지정."""
    if not members:
        return []

    liked_ids = {mid.strip() for mid in liked_member_ids if mid.strip()}
    if not liked_ids:
        return []

    # 성별 필터 결정
    allowed_gender: str | None = None
    if gender_filter == "off":
        allowed_gender = None
    elif gender_filter in ("F", "M"):
        allowed_gender = gender_filter
    else:  # auto
        from collections import Counter
        liked_genders = [m.gender for m in members if m.member_id in liked_ids and m.gender]
        if liked_genders:
            allowed_gender = Counter(liked_genders).most_common(1)[0][0]

    scores = compute_scores(
        primary_members=members,
        liked_member_ids=list(liked_ids),
        secondary_members=secondary_members,
        tertiary_members=tertiary_members,
        weights=weights,
        use_confidence_weight=use_confidence_weight,
    )

    rows: list[dict[str, float | str | int]] = []
    for member in members:
        if member.member_id in liked_ids:
            continue
        if min_image_count > 0 and member.image_count < min_image_count:
            continue
        if min_confidence > 0 and member.confidence < min_confidence:
            continue
        # gender 정보가 있는 멤버에 대해서만 필터 적용. 정보 없으면 통과.
        if allowed_gender and member.gender and member.gender != allowed_gender:
            continue
        entry = scores.get(member.member_id)
        if entry is None:
            continue
        rows.append(
            {
                "member_id": member.member_id,
                "group_name": member.group_name,
                "member_name": member.member_name,
                "image_count": member.image_count,
                "confidence": entry.get("confidence", 0.5),
                "gender": member.gender,
                "score": entry["score"],
                **{k: v for k, v in entry.items() if k.startswith("score_")},
            }
        )

    rows.sort(key=lambda item: float(item["score"]), reverse=True)
    pool = rows[: max(top_k, pool_size)]

    if mmr_lambda >= 1.0 and max_per_group <= 0:
        return pool[:top_k]

    members_by_id = {m.member_id: m for m in members}
    extra_by_id: list[dict[str, MemberVector]] = []
    if secondary_members:
        extra_by_id.append({m.member_id: m for m in secondary_members})
    if tertiary_members:
        extra_by_id.append({m.member_id: m for m in tertiary_members})

    reranked = mmr_rerank(
        candidates=pool,
        members_by_id=members_by_id,
        lambda_relevance=mmr_lambda,
        top_k=top_k,
        max_per_group=max_per_group,
        extra_by_id=extra_by_id or None,
    )
    return reranked


def _format_rows(rows: list[dict[str, str | int | float]]) -> str:
    if not rows:
        return "No recommendations available."

    lines = []
    for index, row in enumerate(rows, start=1):
        extra = " ".join(
            f"{k}={float(v):.2f}" for k, v in row.items() if k.startswith("score_")
        )
        lines.append(
            f"{index:>2}. {row['group_name']} / {row['member_name']} "
            f"({row['member_id']}) score={float(row['score']):.4f} "
            f"conf={float(row.get('confidence', 0.0)):.2f} "
            f"{extra} images={row['image_count']}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Recommend idol members from member vectors.")
    parser.add_argument("--vectors", default="data/member_vectors.csv", help="Primary (예: ArcFace) member vectors.")
    parser.add_argument("--vectors-secondary", default=None, help="Secondary (예: FaRL) member vectors.")
    parser.add_argument("--vectors-tertiary", default=None, help="Tertiary (예: landmark) member vectors.")
    parser.add_argument("--weights", nargs="+", type=float, default=[1.0], help="각 source 의 z-score 합산 가중치.")
    parser.add_argument("--like", nargs="+", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--pool-size", type=int, default=50, help="MMR 적용 전 1차 상위 풀 크기.")
    parser.add_argument("--mmr-lambda", type=float, default=1.0,
                        help="1.0 = 순수 유사도, <1.0 이면 다양성 가산 (0.7 권장).")
    parser.add_argument("--max-per-group", type=int, default=0, help="같은 그룹에서 최대 N명까지 (0=무제한).")
    parser.add_argument("--min-image-count", type=int, default=0, help="추천 후보에 필요한 최소 이미지 수.")
    parser.add_argument("--min-confidence", type=float, default=0.0, help="추천 후보에 필요한 최소 confidence.")
    parser.add_argument("--no-confidence-weight", action="store_true",
                        help="멤버 confidence 로 점수 스케일 조정을 끈다.")
    parser.add_argument(
        "--gender-filter",
        default="auto",
        choices=["auto", "off", "F", "M"],
        help="auto = liked 다수 성별만 노출 (기본), off = 필터 없음, F/M = 명시 지정.",
    )
    args = parser.parse_args()

    primary = load_member_vectors(args.vectors)
    secondary = load_member_vectors(args.vectors_secondary) if args.vectors_secondary else None
    tertiary = load_member_vectors(args.vectors_tertiary) if args.vectors_tertiary else None

    if not primary:
        raise SystemExit(f"No member vectors found at: {args.vectors}")

    rows = recommend_from_members(
        primary,
        liked_member_ids=args.like,
        top_k=args.top_k,
        secondary_members=secondary,
        tertiary_members=tertiary,
        weights=tuple(args.weights),
        use_confidence_weight=not args.no_confidence_weight,
        mmr_lambda=args.mmr_lambda,
        max_per_group=args.max_per_group,
        pool_size=args.pool_size,
        gender_filter=args.gender_filter,
        min_image_count=args.min_image_count,
        min_confidence=args.min_confidence,
    )
    print(_format_rows(rows))


if __name__ == "__main__":
    main()
