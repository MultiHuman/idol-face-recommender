"""Leave-one-out 평가로 ArcFace + FaRL 융합 가중치를 그리드 서치.

평가 지표:
- top-1 accuracy: probe (한 사진) 를 제외한 centroid 를 기준으로, 모든 멤버 centroid 중 자신이 1등인 비율
- mean margin: own_sim - best_other_sim

per-image 임베딩이 양쪽 모두 존재하는 이미지만 사용한다.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT_DIR = Path(__file__).resolve().parent.parent


def _parse_vec(s: str) -> np.ndarray | None:
    s = (s or "").strip()
    if not s:
        return None
    v = np.asarray(json.loads(s), dtype=np.float32)
    if v.ndim != 1 or v.size == 0:
        return None
    return v


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v if n < 1e-12 else v / n


def load_paired_embeddings(
    arc_csv: Path,
    farl_csv: Path,
    min_images: int = 3,
) -> dict[str, dict[str, np.ndarray]]:
    """(member_id, image_path) → {'arc': v, 'farl': v} 인 이미지만 수집."""
    arc: dict[tuple[str, str], np.ndarray] = {}
    with arc_csv.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if r.get("is_valid_face") != "true":
                continue
            mid = (r.get("member_id") or "").strip()
            ip = (r.get("image_path") or "").strip()
            v = _parse_vec(r.get("vector_json") or "")
            if mid and ip and v is not None:
                arc[(mid, ip)] = _normalize(v)

    farl: dict[tuple[str, str], np.ndarray] = {}
    with farl_csv.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            mid = (r.get("member_id") or "").strip()
            ip = (r.get("image_path") or "").strip()
            v = _parse_vec(r.get("farl_vector_json") or "")
            if mid and ip and v is not None:
                farl[(mid, ip)] = _normalize(v)

    pairs: dict[str, list[dict[str, np.ndarray]]] = defaultdict(list)
    for key, va in arc.items():
        vf = farl.get(key)
        if vf is None:
            continue
        pairs[key[0]].append({"arc": va, "farl": vf, "path": key[1]})

    return {mid: items for mid, items in pairs.items() if len(items) >= min_images}


def spherical_mean(vectors: np.ndarray, iters: int = 6) -> np.ndarray:
    if vectors.shape[0] == 1:
        return vectors[0].copy()
    m = vectors.mean(axis=0)
    m /= max(np.linalg.norm(m), 1e-12)
    for _ in range(iters):
        dots = np.clip(vectors @ m, -1.0, 1.0)
        m = m + (vectors - dots[:, None] * m).mean(axis=0)
        m /= max(np.linalg.norm(m), 1e-12)
    return m


def zscore(v: np.ndarray) -> np.ndarray:
    mean = float(v.mean())
    std = float(v.std())
    if std < 1e-6:
        return v - mean
    return (v - mean) / std


def evaluate(
    pairs: dict[str, list[dict[str, np.ndarray]]],
    w_arc: float,
    w_farl: float,
) -> tuple[float, float]:
    """LOO top-1 accuracy 와 mean margin."""
    # 각 멤버별 full centroid (arc, farl)
    arc_centroid: dict[str, np.ndarray] = {}
    farl_centroid: dict[str, np.ndarray] = {}
    arc_stack: dict[str, np.ndarray] = {}
    farl_stack: dict[str, np.ndarray] = {}
    for mid, items in pairs.items():
        a = np.vstack([it["arc"] for it in items])
        f = np.vstack([it["farl"] for it in items])
        arc_stack[mid] = a
        farl_stack[mid] = f
        arc_centroid[mid] = spherical_mean(a)
        farl_centroid[mid] = spherical_mean(f)

    all_ids = list(pairs.keys())
    total, correct = 0, 0
    margins: list[float] = []
    for mid, items in pairs.items():
        a_stack = arc_stack[mid]
        f_stack = farl_stack[mid]
        n = len(items)
        for idx in range(n):
            mask = np.ones(n, dtype=bool)
            mask[idx] = False
            own_arc = spherical_mean(a_stack[mask])
            own_farl = spherical_mean(f_stack[mask])

            # 현재 probe 에 대한 arc/farl 유사도를 전체 멤버에 대해 계산
            arc_sims = np.array(
                [
                    float(items[idx]["arc"] @ (own_arc if other == mid else arc_centroid[other]))
                    for other in all_ids
                ],
                dtype=np.float32,
            )
            farl_sims = np.array(
                [
                    float(items[idx]["farl"] @ (own_farl if other == mid else farl_centroid[other]))
                    for other in all_ids
                ],
                dtype=np.float32,
            )
            # z-score 후 합산
            z_arc = zscore(arc_sims)
            z_farl = zscore(farl_sims)
            fused = w_arc * z_arc + w_farl * z_farl

            own_index = all_ids.index(mid)
            own_score = float(fused[own_index])
            best_other = float(np.delete(fused, own_index).max())

            if own_score > best_other:
                correct += 1
            margins.append(own_score - best_other)
            total += 1

    acc = correct / total if total else 0.0
    return acc, float(np.mean(margins)) if margins else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Grid search fusion weights via LOO top-1.")
    parser.add_argument("--arc", default="data/image_embeddings.csv")
    parser.add_argument("--farl", default="data/image_farl.csv")
    parser.add_argument("--min-images", type=int, default=5)
    parser.add_argument("--grid", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5])
    parser.add_argument(
        "--max-members",
        type=int,
        default=0,
        help="평가에 쓸 멤버 수 제한 (0 = 전체). 그리드 빨리 보고 싶을 때 50~100으로.",
    )
    args = parser.parse_args()

    pairs = load_paired_embeddings(
        ROOT_DIR / args.arc,
        ROOT_DIR / args.farl,
        min_images=args.min_images,
    )
    print(f"Loaded {len(pairs)} members, {sum(len(v) for v in pairs.values())} images (both ArcFace + FaRL present).")

    if args.max_members and args.max_members < len(pairs):
        # 그냥 처음 N명 — 결정적 재현
        subset = dict(list(pairs.items())[: args.max_members])
        pairs = subset
        print(f"Subset to {len(pairs)} members for speed.")

    print(f"\nGrid search over weights {args.grid}:")
    print(f"{'w_arc':>6} {'w_farl':>7} {'acc':>7} {'margin':>8}")
    results: list[tuple[float, float, float, float]] = []
    for wa in args.grid:
        for wf in args.grid:
            if wa == 0 and wf == 0:
                continue
            acc, margin = evaluate(pairs, wa, wf)
            results.append((wa, wf, acc, margin))
            print(f"{wa:6.2f} {wf:7.2f} {acc:7.4f} {margin:8.4f}")

    results.sort(key=lambda x: (-x[2], -x[3]))
    print("\n=== Top 5 by accuracy ===")
    for wa, wf, acc, margin in results[:5]:
        print(f"w_arc={wa:.2f} w_farl={wf:.2f} → acc={acc:.4f}, margin={margin:.4f}")


if __name__ == "__main__":
    main()
