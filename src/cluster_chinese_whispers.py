"""Chinese Whispers 그래프 기반 face clustering.

알고리즘:
1) 멤버 노드 + 코사인 유사도 가중 엣지 그래프 구성
2) 각 노드 라벨 = 자신의 ID로 초기화
3) 각 epoch: 노드들을 무작위 순서로 방문, 이웃의 라벨을 가중치 합산해
   가장 점수가 높은 라벨로 업데이트
4) 수렴할 때까지 반복

Threshold 자동 선택: 평균 cluster 크기가 sqrt(N) 부근이 되도록 스캔.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sklearn.manifold import TSNE


ROOT_DIR = Path(__file__).resolve().parent.parent


def _load_vectors(
    vectors_csv: Path,
    exclude_prefixes: tuple[str, ...] = (),
) -> tuple[list[dict[str, str]], np.ndarray]:
    rows: list[dict[str, str]] = []
    vecs: list[list[float]] = []
    with vectors_csv.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            mid = (r.get("member_id") or "").strip()
            if not mid or any(mid.startswith(p) for p in exclude_prefixes):
                continue
            vec_json = (r.get("vector_json") or "").strip()
            if not vec_json:
                continue
            try:
                v = np.asarray(json.loads(vec_json), dtype=np.float32)
            except json.JSONDecodeError:
                continue
            n = float(np.linalg.norm(v))
            if n > 1e-6:
                v = v / n
            rows.append(r)
            vecs.append(v.tolist())
    return rows, np.asarray(vecs, dtype=np.float32)


def _load_korean_labels(members_csv: Path) -> dict[str, str]:
    labels: dict[str, str] = {}
    if not members_csv.exists():
        return labels
    with members_csv.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            mid = (r.get("member_id") or "").strip()
            hint = (r.get("search_hint") or "").strip()
            if mid and hint:
                labels[mid] = hint
    return labels


def build_similarity_graph(
    matrix: np.ndarray,
    threshold: float,
    k_max: int | None = None,
) -> nx.Graph:
    """행렬 → 코사인 유사도 그래프. threshold 미만은 엣지 없음.
    k_max 가 주어지면 각 노드당 k_max 개 최강 이웃만 유지.
    """
    n = matrix.shape[0]
    sims = matrix @ matrix.T  # cosine (이미 정규화)
    np.fill_diagonal(sims, -1.0)  # self-loop 제거

    G = nx.Graph()
    G.add_nodes_from(range(n))

    if k_max is not None and k_max < n:
        # 각 노드당 top-k 이웃만 유지 → mutual k-NN 그래프
        idx_sorted = np.argsort(-sims, axis=1)  # 내림차순
        kept_edges: set[tuple[int, int]] = set()
        for i in range(n):
            for j in idx_sorted[i, :k_max]:
                if sims[i, j] >= threshold:
                    a, b = (i, j) if i < j else (j, i)
                    kept_edges.add((a, b))
        for i, j in kept_edges:
            G.add_edge(int(i), int(j), weight=float(sims[i, j]))
    else:
        # 임계값만 적용
        for i in range(n):
            for j in range(i + 1, n):
                if sims[i, j] >= threshold:
                    G.add_edge(i, j, weight=float(sims[i, j]))

    return G


def chinese_whispers(
    G: nx.Graph,
    iterations: int = 30,
    seed: int = 42,
) -> dict[int, int]:
    """Chinese Whispers 알고리즘. 노드 → 라벨 dict 리턴."""
    rng = random.Random(seed)
    labels: dict[int, int] = {n: n for n in G.nodes()}

    for it in range(iterations):
        nodes = list(G.nodes())
        rng.shuffle(nodes)
        changed = 0
        for node in nodes:
            neighbor_scores: dict[int, float] = defaultdict(float)
            for nbr in G.neighbors(node):
                w = float(G[node][nbr].get("weight", 1.0))
                neighbor_scores[labels[nbr]] += w
            if not neighbor_scores:
                continue
            new_label = max(neighbor_scores.items(), key=lambda kv: kv[1])[0]
            if labels[node] != new_label:
                labels[node] = new_label
                changed += 1
        if changed == 0:
            break

    return labels


def relabel_clusters(labels: dict[int, int]) -> tuple[dict[int, int], int]:
    """라벨을 0..K-1 로 재매핑. (재매핑된 라벨, 클러스터 수)."""
    unique = sorted(set(labels.values()))
    mapping = {old: new for new, old in enumerate(unique)}
    return {n: mapping[l] for n, l in labels.items()}, len(unique)


def auto_threshold_scan(
    matrix: np.ndarray,
    rows: list[dict[str, str]],
    k_max: int | None,
    iterations: int,
) -> tuple[float, dict[int, int], int]:
    """threshold 스캔 → 가장 적절한 클러스터 수 (target K = sqrt(N))."""
    n_total = matrix.shape[0]
    k_target = max(8, int(round(n_total ** 0.5)))
    print(f"Auto-tuning threshold (target K≈{k_target} for N={n_total})...")

    # ArcFace 의 경우 "다른 사람" 평균 유사도가 0.3~0.4 라 그 이상부터 의미 있음
    candidates: list[tuple[float, int, int, dict[int, int]]] = []
    for thr in [0.40, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.50, 0.52, 0.55]:
        G = build_similarity_graph(matrix, threshold=thr, k_max=k_max)
        n_edges = G.number_of_edges()
        if n_edges == 0:
            print(f"  thr={thr:.2f}: no edges (skip)")
            continue
        labels_dict = chinese_whispers(G, iterations=iterations)
        relabeled, k = relabel_clusters(labels_dict)
        # 클러스터 크기 분포
        sizes = np.bincount(list(relabeled.values()))
        max_size = int(sizes.max())
        # 균등성 점수: max_size 가 너무 크면 페널티
        max_target = max(20, int(n_total / k_target * 1.5))
        balance = 1.0 if max_size <= max_target else max_target / max_size
        # K prior: gaussian
        k_prior = float(np.exp(-((k - k_target) ** 2) / (2 * (k_target / 2) ** 2)))
        score = k_prior * balance
        marker = ""
        candidates.append((thr, k, max_size, relabeled))
        print(f"  thr={thr:.2f}: edges={n_edges:5d}, K={k:3d}, max={max_size:3d}, "
              f"k_prior={k_prior:.2f}, balance={balance:.2f}, score={score:.3f}")

    if not candidates:
        raise SystemExit("No valid threshold found.")

    # combined score 기준 정렬
    def _score(t: tuple[float, int, int, dict[int, int]]) -> float:
        thr, k, max_size, _ = t
        max_target = max(20, int(n_total / k_target * 1.5))
        balance = 1.0 if max_size <= max_target else max_target / max_size
        k_prior = float(np.exp(-((k - k_target) ** 2) / (2 * (k_target / 2) ** 2)))
        return k_prior * balance

    candidates.sort(key=_score, reverse=True)
    best_thr, best_k, best_max, best_labels = candidates[0]
    print(f"\nSelected threshold={best_thr:.2f} → {best_k} clusters, max={best_max}")
    return best_thr, best_labels, best_k


def visualize(
    matrix: np.ndarray,
    labels: dict[int, int],
    output_png: Path,
    title: str,
) -> None:
    print("Computing t-SNE for visualization...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, max(5, matrix.shape[0] // 10)), init="pca")
    coords = tsne.fit_transform(matrix)

    cluster_ids = np.array([labels[i] for i in range(matrix.shape[0])])
    unique = sorted(set(cluster_ids.tolist()))
    cmap = plt.get_cmap("tab20")

    fig, ax = plt.subplots(figsize=(16, 12))
    for cid in unique:
        mask = cluster_ids == cid
        color = cmap(cid % 20)
        ax.scatter(coords[mask, 0], coords[mask, 1], c=[color], s=30, alpha=0.8,
                   label=f"c{cid} ({mask.sum()})")
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    if len(unique) <= 30:
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8, ncol=1)
    plt.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=120, bbox_inches="tight")
    print(f"Saved {output_png}")
    return coords


def run(
    vectors_csv: Path,
    members_csv: Path,
    output_csv: Path,
    output_png: Path,
    threshold: float | None,
    k_max: int | None,
    iterations: int,
    exclude_prefixes: tuple[str, ...],
) -> None:
    rows, matrix = _load_vectors(vectors_csv, exclude_prefixes=exclude_prefixes)
    if len(rows) == 0:
        raise SystemExit(f"No vectors loaded from {vectors_csv}")
    print(f"Loaded {len(rows)} member vectors of dim {matrix.shape[1]}")
    if exclude_prefixes:
        print(f"Excluded prefixes: {list(exclude_prefixes)}")

    if threshold is None:
        threshold, labels, n_clusters = auto_threshold_scan(matrix, rows, k_max, iterations)
    else:
        G = build_similarity_graph(matrix, threshold=threshold, k_max=k_max)
        print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        labels_dict = chinese_whispers(G, iterations=iterations)
        labels, n_clusters = relabel_clusters(labels_dict)
        print(f"Chinese Whispers: {n_clusters} clusters")

    ko_labels = _load_korean_labels(members_csv)
    coords = visualize(matrix, labels, output_png, title=f"Chinese Whispers (thr={threshold:.2f}, K={n_clusters})")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["member_id", "label", "cluster_id", "x", "y"])
        w.writeheader()
        for i, row in enumerate(rows):
            w.writerow({
                "member_id": row["member_id"],
                "label": ko_labels.get(row["member_id"], row["member_id"]),
                "cluster_id": labels[i],
                "x": f"{coords[i, 0]:.4f}",
                "y": f"{coords[i, 1]:.4f}",
            })
    print(f"Wrote {output_csv}")

    # 클러스터 요약
    per_cluster: dict[int, list[str]] = defaultdict(list)
    for i, row in enumerate(rows):
        cid = labels[i]
        per_cluster[cid].append(ko_labels.get(row["member_id"], row["member_id"]))
    print()
    print(f"=== {n_clusters} clusters ===")
    for cid, members in sorted(per_cluster.items(), key=lambda x: -len(x[1])):
        sample = ", ".join(members[:8])
        if len(members) > 8:
            sample += f", ... (+{len(members) - 8})"
        print(f"c{cid} ({len(members)}명): {sample}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chinese Whispers face clustering.")
    parser.add_argument("--vectors", default="data/member_vectors.csv")
    parser.add_argument("--members", default="data/members.csv")
    parser.add_argument("--output-csv", default="data/cluster_cw.csv")
    parser.add_argument("--output-png", default="data/face_clusters_cw.png")
    parser.add_argument("--threshold", type=float, default=None,
                        help="코사인 임계값 (생략시 자동 스캔)")
    parser.add_argument("--k-max", type=int, default=20,
                        help="각 노드당 top-k 이웃만 유지 (mutual kNN). 0 또는 음수면 비활성화.")
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--exclude-group-prefix", nargs="*", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    k_max = args.k_max if args.k_max and args.k_max > 0 else None
    exclude_prefixes = tuple(f"{p}__" for p in args.exclude_group_prefix)
    run(
        vectors_csv=ROOT_DIR / args.vectors,
        members_csv=ROOT_DIR / args.members,
        output_csv=ROOT_DIR / args.output_csv,
        output_png=ROOT_DIR / args.output_png,
        threshold=args.threshold,
        k_max=k_max,
        iterations=args.iterations,
        exclude_prefixes=exclude_prefixes,
    )


if __name__ == "__main__":
    main()
