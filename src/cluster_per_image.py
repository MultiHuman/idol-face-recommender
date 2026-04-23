"""사진 단위 ArcFace 임베딩에 직접 클러스터링 → 멤버 단위 majority vote.

평균 벡터를 쓰지 않고 모든 유효 사진을 점으로 사용해서 UMAP+HDBSCAN.
멤버에게는 그 멤버 사진들의 majority cluster 를 할당.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

try:
    import umap  # type: ignore
    _UMAP_AVAILABLE = True
except ImportError:
    _UMAP_AVAILABLE = False


ROOT_DIR = Path(__file__).resolve().parent.parent


def _load_image_vectors(
    embeddings_csv: Path,
    exclude_prefixes: tuple[str, ...] = (),
) -> tuple[list[dict[str, str]], np.ndarray]:
    rows: list[dict[str, str]] = []
    vecs: list[list[float]] = []
    with embeddings_csv.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if r.get("is_valid_face") != "true":
                continue
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


def auto_select_hdbscan(
    reduced: np.ndarray,
    target_k: int,
    min_samples: int = 3,
) -> tuple[np.ndarray, int]:
    """HDBSCAN min_cluster_size + cluster_selection_method 자동 선택."""
    n_total = len(reduced)
    k_sigma = max(5.0, target_k / 2)
    max_size_target = max(50, int(n_total / target_k * 2))
    print(f"Auto-tuning HDBSCAN (target K≈{target_k}, max≈{max_size_target} for N={n_total})...")

    best_score = -1.0
    best_labels = None
    best_info = None
    scan_range = list(range(5, min(60, n_total // 30 + 1), 3))
    for csm in ("eom", "leaf"):
        for mcs in scan_range:
            clust = HDBSCAN(min_cluster_size=mcs, min_samples=min_samples,
                            metric="euclidean", cluster_selection_method=csm)
            labels = clust.fit_predict(reduced)
            n_cl = int(labels.max() + 1) if labels.max() >= 0 else 0
            n_ns = int((labels == -1).sum())
            noise_frac = n_ns / n_total
            non_noise = labels != -1
            if n_cl < 2 or non_noise.sum() < 20:
                continue
            try:
                # silhouette: 큰 데이터에서 sample
                sil = silhouette_score(
                    reduced[non_noise], labels[non_noise],
                    metric="euclidean",
                    sample_size=min(non_noise.sum(), 1000),
                    random_state=42,
                )
            except ValueError:
                continue
            sizes = np.bincount(labels[non_noise])
            max_size = int(sizes.max())
            balance = 1.0 if max_size <= max_size_target else max_size_target / max_size
            k_prior = float(np.exp(-((n_cl - target_k) ** 2) / (2 * k_sigma * k_sigma)))
            noise_factor = (1.0 - noise_frac) ** 0.5
            combined = sil * noise_factor * k_prior * balance
            marker = ""
            if combined > best_score:
                best_score = combined
                best_labels = labels
                best_info = (csm, mcs, n_cl, n_ns, sil, max_size)
                marker = " ★"
            print(f"  csm={csm:4s} mcs={mcs:2d}: K={n_cl:3d}, noise={noise_frac:.0%}, "
                  f"max={max_size:4d}, sil={sil:.3f}, k_prior={k_prior:.2f}, "
                  f"balance={balance:.2f}, combined={combined:.4f}{marker}")
    if best_labels is None:
        raise SystemExit("No valid HDBSCAN result found.")
    csm, mcs, n_cl, n_ns, sil, max_size = best_info
    print(f"Selected csm={csm}, mcs={mcs} → {n_cl} clusters, {n_ns} noise, max={max_size}")
    return best_labels, n_cl


def aggregate_member_clusters(
    rows: list[dict[str, str]],
    image_clusters: np.ndarray,
) -> dict[str, tuple[int, int, int, float]]:
    """멤버별 majority cluster + 통계.

    리턴: {member_id: (best_cluster, count, total_photos, consensus_ratio)}
    """
    per_member: dict[str, list[int]] = defaultdict(list)
    for row, cid in zip(rows, image_clusters):
        per_member[row["member_id"]].append(int(cid))

    result: dict[str, tuple[int, int, int, float]] = {}
    for mid, cids in per_member.items():
        # noise (-1) 제외하고 majority
        non_noise = [c for c in cids if c >= 0]
        total = len(cids)
        if non_noise:
            counter = Counter(non_noise)
            best_cid, best_count = counter.most_common(1)[0]
            ratio = best_count / total
        else:
            best_cid = -1
            best_count = 0
            ratio = 0.0
        result[mid] = (best_cid, best_count, total, ratio)
    return result


def visualize(
    matrix: np.ndarray,
    cluster_ids: np.ndarray,
    output_png: Path,
    title: str,
    sample: int = 3000,
) -> np.ndarray | None:
    print("Computing t-SNE for visualization (sampled)...")
    n = matrix.shape[0]
    if n > sample:
        idx = np.random.RandomState(42).choice(n, sample, replace=False)
        sub_matrix = matrix[idx]
        sub_clusters = cluster_ids[idx]
    else:
        idx = np.arange(n)
        sub_matrix = matrix
        sub_clusters = cluster_ids

    tsne = TSNE(n_components=2, random_state=42, perplexity=30, init="pca")
    coords = tsne.fit_transform(sub_matrix)

    unique = sorted(set(sub_clusters.tolist()))
    cmap = plt.get_cmap("tab20")
    fig, ax = plt.subplots(figsize=(16, 12))
    for cid in unique:
        mask = sub_clusters == cid
        if cid == -1:
            ax.scatter(coords[mask, 0], coords[mask, 1], c="lightgray", s=8, alpha=0.4,
                       label=f"noise ({mask.sum()})")
        else:
            color = cmap(cid % 20)
            ax.scatter(coords[mask, 0], coords[mask, 1], c=[color], s=10, alpha=0.7,
                       label=f"c{cid} ({mask.sum()})" if mask.sum() >= 30 else None)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7, ncol=1)
    plt.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=120, bbox_inches="tight")
    print(f"Saved {output_png}")
    return None


def run(
    embeddings_csv: Path,
    members_csv: Path,
    output_csv: Path,
    output_png: Path,
    pca_dim: int,
    umap_n_components: int,
    umap_n_neighbors: int,
    umap_min_dist: float,
    target_k: int,
    exclude_prefixes: tuple[str, ...],
) -> None:
    if not _UMAP_AVAILABLE:
        raise SystemExit("umap-learn not installed.")

    print(f"Loading per-image ArcFace vectors from {embeddings_csv}...")
    rows, matrix = _load_image_vectors(embeddings_csv, exclude_prefixes=exclude_prefixes)
    print(f"Loaded {len(rows)} valid face images")
    if exclude_prefixes:
        print(f"Excluded prefixes: {list(exclude_prefixes)}")

    # PCA
    pca_dim_eff = min(pca_dim, matrix.shape[1], matrix.shape[0])
    pca = PCA(n_components=pca_dim_eff, random_state=42)
    reduced = pca.fit_transform(matrix)
    print(f"PCA: {matrix.shape[1]}D → {pca_dim_eff}D ({pca.explained_variance_ratio_.sum():.2%} variance)")
    norms = np.linalg.norm(reduced, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    reduced = reduced / norms

    # UMAP
    print(f"UMAP: {pca_dim_eff}D → {umap_n_components}D "
          f"(n_neighbors={umap_n_neighbors}, min_dist={umap_min_dist})")
    reducer = umap.UMAP(
        n_components=umap_n_components,
        n_neighbors=umap_n_neighbors,
        min_dist=umap_min_dist,
        metric="cosine",
        random_state=42,
    )
    reduced = reducer.fit_transform(reduced)
    print(f"UMAP done. Shape: {reduced.shape}")

    # HDBSCAN auto
    image_clusters, n_clusters = auto_select_hdbscan(reduced, target_k=target_k)

    # 멤버별 majority vote
    member_assign = aggregate_member_clusters(rows, image_clusters)
    print(f"\nAggregated to {len(member_assign)} members.")

    # consensus 통계
    ratios = [r for _, _, _, r in member_assign.values() if r > 0]
    if ratios:
        print(f"Consensus ratio: mean={np.mean(ratios):.2f}, "
              f"min={np.min(ratios):.2f}, max={np.max(ratios):.2f}")
    n_noise_member = sum(1 for c, _, _, _ in member_assign.values() if c == -1)
    print(f"Members with no cluster: {n_noise_member}")

    ko_labels = _load_korean_labels(members_csv)

    # CSV 저장 (멤버 단위)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["member_id", "label", "cluster_id",
                                           "majority_count", "total_photos", "consensus"])
        w.writeheader()
        for mid, (cid, count, total, ratio) in sorted(member_assign.items()):
            w.writerow({
                "member_id": mid,
                "label": ko_labels.get(mid, mid),
                "cluster_id": cid,
                "majority_count": count,
                "total_photos": total,
                "consensus": f"{ratio:.3f}",
            })
    print(f"Wrote {output_csv}")

    # 시각화 (사진 단위)
    visualize(reduced, image_clusters, output_png,
              title=f"Per-image Chinese face clusters ({n_clusters}, N={len(rows)})")

    # 클러스터 요약
    per_cluster: dict[int, list[str]] = defaultdict(list)
    for mid, (cid, _, _, _) in member_assign.items():
        per_cluster[cid].append(ko_labels.get(mid, mid))
    print()
    print(f"=== {n_clusters} clusters (member-level after majority vote) ===")
    for cid, members in sorted(per_cluster.items(), key=lambda x: (-len(x[1]), x[0])):
        if cid == -1:
            print(f"noise ({len(members)}명): {', '.join(members[:5])}{'...' if len(members)>5 else ''}")
            continue
        sample = ", ".join(members[:8])
        if len(members) > 8:
            sample += f", ... (+{len(members) - 8})"
        print(f"c{cid} ({len(members)}명): {sample}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Per-image clustering with member-level majority vote.")
    parser.add_argument("--embeddings", default="data/image_embeddings.csv")
    parser.add_argument("--members", default="data/members.csv")
    parser.add_argument("--output-csv", default="data/cluster_per_image.csv")
    parser.add_argument("--output-png", default="data/face_clusters_per_image.png")
    parser.add_argument("--pca-dim", type=int, default=100)
    parser.add_argument("--umap-n-components", type=int, default=10)
    parser.add_argument("--umap-n-neighbors", type=int, default=30)
    parser.add_argument("--umap-min-dist", type=float, default=0.0)
    parser.add_argument("--target-k", type=int, default=25,
                        help="Approximate target number of clusters")
    parser.add_argument("--exclude-group-prefix", nargs="*", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exclude_prefixes = tuple(f"{p}__" for p in args.exclude_group_prefix)
    run(
        embeddings_csv=ROOT_DIR / args.embeddings,
        members_csv=ROOT_DIR / args.members,
        output_csv=ROOT_DIR / args.output_csv,
        output_png=ROOT_DIR / args.output_png,
        pca_dim=args.pca_dim,
        umap_n_components=args.umap_n_components,
        umap_n_neighbors=args.umap_n_neighbors,
        umap_min_dist=args.umap_min_dist,
        target_k=args.target_k,
        exclude_prefixes=exclude_prefixes,
    )


if __name__ == "__main__":
    main()
