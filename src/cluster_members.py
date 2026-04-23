"""멤버 벡터를 클러스터링하고 2D로 시각화."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import HDBSCAN, KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

try:
    import umap  # type: ignore
    _UMAP_AVAILABLE = True
except ImportError:
    _UMAP_AVAILABLE = False


ROOT_DIR = Path(__file__).resolve().parent.parent


def _load_vectors(path: Path) -> tuple[list[dict[str, str]], np.ndarray]:
    rows: list[dict[str, str]] = []
    vectors: list[list[float]] = []
    with path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            vec_json = (row.get("vector_json") or "").strip()
            if not vec_json:
                continue
            try:
                vec = json.loads(vec_json)
            except json.JSONDecodeError:
                continue
            if not vec:
                continue
            rows.append(row)
            vectors.append(vec)
    return rows, np.asarray(vectors, dtype=np.float32)


def _load_farl_features(
    farl_csv: Path,
    vectors_csv: Path,
    min_images: int = 3,
) -> tuple[list[dict[str, str]], np.ndarray]:
    """per-image FaRL 임베딩 → member 평균 → (rows, matrix)."""
    from collections import defaultdict
    per_member: dict[str, list[list[float]]] = defaultdict(list)
    with farl_csv.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            mid = (row.get("member_id") or "").strip()
            vec_json = (row.get("farl_vector_json") or "").strip()
            if not mid or not vec_json:
                continue
            try:
                vec = json.loads(vec_json)
            except json.JSONDecodeError:
                continue
            per_member[mid].append(vec)

    rows: list[dict[str, str]] = []
    vectors: list[list[float]] = []
    with vectors_csv.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            mid = (row.get("member_id") or "").strip()
            images = per_member.get(mid, [])
            if len(images) < min_images:
                continue
            mean_vec = np.mean(images, axis=0)
            # L2 normalize after averaging (cosine space)
            norm = float(np.linalg.norm(mean_vec))
            if norm > 1e-6:
                mean_vec = mean_vec / norm
            rows.append(row)
            vectors.append(mean_vec.tolist())

    return rows, np.asarray(vectors, dtype=np.float32)


def _load_landmark_features(
    landmarks_csv: Path,
    vectors_csv: Path,
    min_images: int = 3,
) -> tuple[list[dict[str, str]], np.ndarray]:
    """per-image landmark features → member 평균 → (rows, matrix)."""
    # 멤버별 이미지 feature 수집
    from collections import defaultdict
    feature_names: list[str] = []
    per_member: dict[str, list[list[float]]] = defaultdict(list)
    with landmarks_csv.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        feature_names = [name for name in (reader.fieldnames or []) if name not in ("member_id", "image_path")]
        for row in reader:
            mid = (row.get("member_id") or "").strip()
            if not mid:
                continue
            try:
                vec = [float(row[name]) for name in feature_names]
            except (ValueError, KeyError):
                continue
            per_member[mid].append(vec)

    # member_vectors.csv 로 순서 결정 + 메타데이터
    rows: list[dict[str, str]] = []
    vectors: list[list[float]] = []
    with vectors_csv.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            mid = (row.get("member_id") or "").strip()
            images = per_member.get(mid, [])
            if len(images) < min_images:
                continue
            mean_vec = np.mean(images, axis=0).tolist()
            rows.append(row)
            vectors.append(mean_vec)

    matrix = np.asarray(vectors, dtype=np.float32)
    return rows, matrix


def _load_korean_labels(members_csv: Path) -> dict[str, str]:
    labels: dict[str, str] = {}
    if not members_csv.exists():
        return labels
    with members_csv.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            mid = (row.get("member_id") or "").strip()
            hint = (row.get("search_hint") or "").strip()
            if mid and hint:
                labels[mid] = hint
    return labels


def run(
    vectors_csv: Path,
    members_csv: Path,
    output_csv: Path,
    output_png: Path,
    min_cluster_size: int,
    min_samples: int,
    pca_dim: int,
    method: str,
    n_clusters: int,
    source: str = "arcface",
    landmarks_csv: Path | None = None,
    farl_csv: Path | None = None,
    exclude_group_prefix: list[str] | None = None,
    umap_n_components: int = 15,
    umap_n_neighbors: int = 30,
    umap_min_dist: float = 0.0,
) -> None:
    excluded_prefixes = tuple(f"{p}__" for p in (exclude_group_prefix or []))
    if source == "farl":
        if farl_csv is None or not farl_csv.exists():
            raise SystemExit(f"FaRL CSV not found: {farl_csv}")
        rows, matrix = _load_farl_features(farl_csv, vectors_csv)
        if len(rows) == 0:
            raise SystemExit("No FaRL features loaded.")
        if excluded_prefixes:
            keep = [i for i, r in enumerate(rows) if not r["member_id"].startswith(excluded_prefixes)]
            rows = [rows[i] for i in keep]
            matrix = matrix[keep]
            print(f"Excluded groups {list(exclude_group_prefix)}: kept {len(rows)} members")
        print(f"Loaded {len(rows)} members with {matrix.shape[1]}D FaRL embeddings")

        # FaRL도 ArcFace처럼 PCA로 차원 축소
        pca_dim_eff = min(pca_dim, matrix.shape[1], matrix.shape[0])
        pca = PCA(n_components=pca_dim_eff, random_state=42)
        reduced = pca.fit_transform(matrix)
        print(f"PCA: {matrix.shape[1]}D → {pca_dim_eff}D (explained variance: {pca.explained_variance_ratio_.sum():.2%})")

        # PCA 후 재정규화 (unit vector로 복원) — cosine 기반 모델이므로 spherical space에서 클러스터링
        norms = np.linalg.norm(reduced, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        reduced = reduced / norms
        print("Renormalized PCA output to unit vectors (spherical space)")
    elif source == "landmark":
        if landmarks_csv is None or not landmarks_csv.exists():
            raise SystemExit(f"Landmarks CSV not found: {landmarks_csv}")
        rows, matrix = _load_landmark_features(landmarks_csv, vectors_csv)
        if len(rows) == 0:
            raise SystemExit("No landmark features loaded.")
        if excluded_prefixes:
            keep = [i for i, r in enumerate(rows) if not r["member_id"].startswith(excluded_prefixes)]
            rows = [rows[i] for i in keep]
            matrix = matrix[keep]
            print(f"Excluded groups {list(exclude_group_prefix)}: kept {len(rows)} members")
        print(f"Loaded {len(rows)} members with {matrix.shape[1]} landmark features")

        # 기하학 특징은 차원이 다르므로 StandardScaler로 정규화
        scaler = StandardScaler()
        reduced = scaler.fit_transform(matrix)
        print(f"StandardScaler applied to {matrix.shape[1]}D landmark features")
    else:
        rows, matrix = _load_vectors(vectors_csv)
        if len(rows) == 0:
            raise SystemExit(f"No vectors in {vectors_csv}")
        if excluded_prefixes:
            keep = [i for i, r in enumerate(rows) if not r["member_id"].startswith(excluded_prefixes)]
            rows = [rows[i] for i in keep]
            matrix = matrix[keep]
            print(f"Excluded groups {list(exclude_group_prefix)}: kept {len(rows)} members")

        print(f"Loaded {len(rows)} member vectors of dim {matrix.shape[1]}")

        # L2 정규화 (face recognition embedding은 보통 정규화 후 사용)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        matrix_norm = matrix / norms

        # 1) PCA로 차원 축소 (클러스터링 안정성 & 속도)
        pca_dim_eff = min(pca_dim, matrix_norm.shape[1], matrix_norm.shape[0])
        pca = PCA(n_components=pca_dim_eff, random_state=42)
        reduced = pca.fit_transform(matrix_norm)
        print(f"PCA: {matrix_norm.shape[1]}D → {pca_dim_eff}D (explained variance: {pca.explained_variance_ratio_.sum():.2%})")

        # PCA 후 재정규화 — ArcFace는 cosine 기반이므로 unit sphere 복원
        norms = np.linalg.norm(reduced, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        reduced = reduced / norms
        print("Renormalized PCA output to unit vectors (spherical space)")

    # 2) UMAP 중간 단계 (선택적)
    if method in ("umap-hdbscan", "auto-umap"):
        if not _UMAP_AVAILABLE:
            raise SystemExit(
                "umap-learn 이 설치되지 않았어. 설치: pip install umap-learn"
            )
        print(f"UMAP: {reduced.shape[1]}D → {umap_n_components}D "
              f"(n_neighbors={umap_n_neighbors}, min_dist={umap_min_dist}, metric=cosine)")
        reducer = umap.UMAP(
            n_components=umap_n_components,
            n_neighbors=umap_n_neighbors,
            min_dist=umap_min_dist,
            metric="cosine",
            random_state=42,
        )
        reduced = reducer.fit_transform(reduced)
        print(f"UMAP done. Running HDBSCAN on {reduced.shape}")

        if method == "auto-umap":
            # min_cluster_size × cluster_selection_method 스캔
            # 점수: silhouette × (1-noise) × K_prior × balance
            # K_prior: K=sqrt(N) 근처 가우시안
            # balance: max_cluster_size 가 too dominant 하면 페널티
            n_total = len(reduced)
            k_target = max(8, int(round(n_total ** 0.5)))
            k_sigma = max(5.0, k_target / 2)
            max_size_target = max(20, int(n_total / k_target * 1.5))
            print(f"Auto-selecting (target K≈{k_target}, max_cluster≈{max_size_target} for N={n_total})...")
            scan_range = list(range(3, min(31, n_total // 8 + 1)))
            best_score = -1.0
            best_params = None
            best_labels = None
            best_stats = (0, 0, 0.0, 0)
            for csm in ("eom", "leaf"):
                for mcs in scan_range:
                    clust = HDBSCAN(
                        min_cluster_size=mcs,
                        min_samples=min_samples,
                        metric="euclidean",
                        cluster_selection_method=csm,
                    )
                    labels = clust.fit_predict(reduced)
                    n_cl = int(labels.max() + 1) if labels.max() >= 0 else 0
                    n_ns = int((labels == -1).sum())
                    noise_frac = n_ns / n_total
                    non_noise_mask = labels != -1
                    if n_cl < 2 or non_noise_mask.sum() < 10:
                        continue
                    try:
                        sil = silhouette_score(reduced[non_noise_mask], labels[non_noise_mask], metric="euclidean")
                    except ValueError:
                        continue
                    sizes = np.bincount(labels[non_noise_mask])
                    max_size = int(sizes.max())
                    # 균등성 페널티: 최대 클러스터 크기가 target 초과시 penalty
                    if max_size <= max_size_target:
                        balance = 1.0
                    else:
                        balance = max_size_target / max_size  # 선형 감소
                    k_prior = float(np.exp(-((n_cl - k_target) ** 2) / (2 * k_sigma * k_sigma)))
                    noise_factor = (1.0 - noise_frac) ** 0.5
                    combined = sil * noise_factor * k_prior * balance
                    marker = ""
                    if combined > best_score:
                        best_score = combined
                        best_params = (csm, mcs)
                        best_labels = labels
                        best_stats = (n_cl, n_ns, sil, max_size)
                        marker = " ★"
                    print(f"  csm={csm:4s} mcs={mcs:2d}: K={n_cl:3d}, noise={noise_frac:.0%}, "
                          f"max={max_size:3d}, sil={sil:.3f}, balance={balance:.2f}, "
                          f"combined={combined:.4f}{marker}")
            if best_labels is None:
                raise SystemExit("No valid HDBSCAN result found.")
            cluster_ids = best_labels
            n_clusters_found, n_noise, sil, max_size = best_stats
            print(f"Selected csm={best_params[0]}, mcs={best_params[1]} → "
                  f"{n_clusters_found} clusters, {n_noise} noise ({n_noise/n_total:.1%}), "
                  f"max_size={max_size}, silhouette={sil:.4f}")
        else:
            clusterer = HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric="euclidean",
            )
            cluster_ids = clusterer.fit_predict(reduced)
            n_clusters_found = int(cluster_ids.max() + 1) if cluster_ids.max() >= 0 else 0
            n_noise = int((cluster_ids == -1).sum())
            print(f"UMAP+HDBSCAN: {n_clusters_found} clusters, {n_noise} noise ({n_noise/len(reduced):.1%})")
    # 3) 다른 클러스터링 방법
    elif method == "auto" or method == "auto-cosine":
        # k=2~20 중 실루엣 점수 최대인 k 자동 선택
        best_k = 2
        best_score = -1.0
        best_labels = None
        use_cosine = method == "auto-cosine"
        silhouette_metric = "cosine" if use_cosine else "euclidean"
        print(f"Auto-selecting k via silhouette score (metric={silhouette_metric})...")
        for k in range(2, 21):
            if use_cosine:
                clust = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average")
            else:
                clust = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = clust.fit_predict(reduced)
            score = silhouette_score(reduced, labels, metric=silhouette_metric, sample_size=min(len(reduced), 500), random_state=42)
            print(f"  k={k}: silhouette={score:.4f}")
            if score > best_score:
                best_score = score
                best_k = k
                best_labels = labels
        cluster_ids = best_labels
        n_noise = 0
        n_clusters_found = best_k
        print(f"Best k = {best_k} (silhouette = {best_score:.4f})")
    elif method == "agglomerative":
        clusterer = AgglomerativeClustering(n_clusters=n_clusters, metric="cosine", linkage="average")
        cluster_ids = clusterer.fit_predict(reduced)
        n_noise = 0
        n_clusters_found = n_clusters
        print(f"Agglomerative(cosine): {n_clusters_found} clusters")
    elif method == "kmeans":
        clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_ids = clusterer.fit_predict(reduced)
        n_noise = 0
        n_clusters_found = n_clusters
        print(f"KMeans: {n_clusters_found} clusters")
    else:
        clusterer = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="euclidean",
        )
        cluster_ids = clusterer.fit_predict(reduced)
        n_clusters_found = int(cluster_ids.max() + 1) if cluster_ids.max() >= 0 else 0
        n_noise = int((cluster_ids == -1).sum())
        print(f"HDBSCAN: {n_clusters_found} clusters, {n_noise} noise points ({n_noise/len(rows):.1%})")

    # 3) t-SNE 2D 시각화
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, max(5, len(rows) // 10)), init="pca")
    coords_2d = tsne.fit_transform(reduced)
    print(f"t-SNE done.")

    # 4) 클러스터별 집계
    ko_labels = _load_korean_labels(members_csv)
    per_cluster: dict[int, list[str]] = {}
    for i, row in enumerate(rows):
        cid = int(cluster_ids[i])
        per_cluster.setdefault(cid, []).append(ko_labels.get(row["member_id"], row["member_id"]))

    # 5) CSV 저장
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["member_id", "label", "cluster_id", "x", "y"])
        writer.writeheader()
        for i, row in enumerate(rows):
            writer.writerow({
                "member_id": row["member_id"],
                "label": ko_labels.get(row["member_id"], row["member_id"]),
                "cluster_id": int(cluster_ids[i]),
                "x": f"{coords_2d[i, 0]:.4f}",
                "y": f"{coords_2d[i, 1]:.4f}",
            })
    print(f"Wrote {output_csv}")

    # 6) 시각화 — 클러스터별 색상
    fig, ax = plt.subplots(figsize=(16, 12))
    unique_clusters = sorted(set(int(c) for c in cluster_ids))
    cmap = plt.get_cmap("tab20")
    for cid in unique_clusters:
        mask = cluster_ids == cid
        if cid == -1:
            ax.scatter(
                coords_2d[mask, 0],
                coords_2d[mask, 1],
                c="lightgray",
                s=15,
                alpha=0.5,
                label=f"noise ({mask.sum()})",
            )
        else:
            color = cmap(cid % 20)
            ax.scatter(
                coords_2d[mask, 0],
                coords_2d[mask, 1],
                c=[color],
                s=30,
                alpha=0.8,
                label=f"c{cid} ({mask.sum()})",
            )
    ax.set_title(
        f"Idol face clusters ({method}) — {n_clusters_found} clusters, {n_noise} noise (of {len(rows)} members)",
        fontsize=14,
    )
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8, ncol=1)
    plt.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=120, bbox_inches="tight")
    print(f"Wrote {output_png}")

    # 7) 클러스터 요약 출력 (샘플 멤버)
    print("\n=== Cluster summary ===")
    for cid in sorted(per_cluster.keys()):
        if cid == -1:
            continue
        members = per_cluster[cid]
        sample = ", ".join(members[:8])
        if len(members) > 8:
            sample += f", ... (+{len(members) - 8})"
        print(f"c{cid} ({len(members)}명): {sample}")
    if -1 in per_cluster:
        print(f"noise ({len(per_cluster[-1])}명): {', '.join(per_cluster[-1][:5])}...")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cluster idol member face vectors.")
    parser.add_argument("--vectors", default="data/member_vectors.csv")
    parser.add_argument("--members", default="data/members.csv")
    parser.add_argument("--output-csv", default="data/cluster_assignments.csv")
    parser.add_argument("--output-png", default="data/face_clusters.png")
    parser.add_argument("--source", choices=["arcface", "landmark", "farl"], default="arcface")
    parser.add_argument("--landmarks", default="data/image_landmarks.csv")
    parser.add_argument("--farl", default="data/image_farl.csv")
    parser.add_argument(
        "--exclude-group-prefix",
        nargs="*",
        default=[],
        help="제외할 그룹 prefix (예: aespa ive nmixx). member_id가 '{prefix}__' 로 시작하면 제외.",
    )
    parser.add_argument(
        "--method",
        choices=["auto", "auto-cosine", "kmeans", "agglomerative", "hdbscan", "umap-hdbscan", "auto-umap"],
        default="auto",
    )
    parser.add_argument("--umap-n-components", type=int, default=15, help="UMAP 출력 차원")
    parser.add_argument("--umap-n-neighbors", type=int, default=30, help="UMAP n_neighbors (클러스터링 용 30+)")
    parser.add_argument("--umap-min-dist", type=float, default=0.0, help="UMAP min_dist (0.0 권장)")
    parser.add_argument("--n-clusters", type=int, default=12, help="KMeans only")
    parser.add_argument("--min-cluster-size", type=int, default=8, help="HDBSCAN only")
    parser.add_argument("--min-samples", type=int, default=3, help="HDBSCAN only")
    parser.add_argument("--pca-dim", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(
        vectors_csv=ROOT_DIR / args.vectors,
        members_csv=ROOT_DIR / args.members,
        output_csv=ROOT_DIR / args.output_csv,
        output_png=ROOT_DIR / args.output_png,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        pca_dim=args.pca_dim,
        method=args.method,
        n_clusters=args.n_clusters,
        source=args.source,
        landmarks_csv=ROOT_DIR / args.landmarks,
        farl_csv=ROOT_DIR / args.farl,
        exclude_group_prefix=args.exclude_group_prefix,
        umap_n_components=args.umap_n_components,
        umap_n_neighbors=args.umap_n_neighbors,
        umap_min_dist=args.umap_min_dist,
    )


if __name__ == "__main__":
    main()
