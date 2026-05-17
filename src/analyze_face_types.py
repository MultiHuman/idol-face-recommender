"""Explore member-level face-type clusters.

The analysis is intentionally exploratory:
- collapse known cross-group aliases into one person
- split by gender so clusters are not dominated by the obvious gender axis
- use ArcFace face-recognition vectors by default, extracted from aligned face crops
- cluster with cosine agglomerative clustering
- write a Markdown report, CSV assignments, and local contact sheets
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from src.recommend import MemberVector, load_member_aliases, load_member_vectors


ROOT_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class FaceTypeMember:
    member: MemberVector
    alias_id: str
    label: str
    feature: np.ndarray


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return vector
    return vector / norm


def _clean_label(text: str) -> str:
    text = re.sub(r'["]', "", text or "")
    return re.sub(r"\s+", " ", text).strip()


def load_labels(members_csv: Path) -> dict[str, str]:
    labels: dict[str, str] = {}
    if not members_csv.exists():
        return labels
    with members_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            member_id = (row.get("member_id") or "").strip()
            hint = _clean_label(row.get("search_hint") or "")
            if member_id and hint:
                labels[member_id] = hint
    return labels


def quality_rank(member: MemberVector) -> tuple[float, int, str]:
    return (member.confidence, member.image_count, member.member_id)


def build_feature(
    member: MemberVector,
    farl_member: MemberVector | None,
    feature_mode: str,
    farl_weight: float,
) -> np.ndarray | None:
    arcface = _normalize(member.vector.astype(np.float32))
    if feature_mode == "arcface":
        return arcface

    if farl_member is None:
        return None

    farl = _normalize(farl_member.vector.astype(np.float32))
    if feature_mode == "farl":
        return farl

    return _normalize(np.concatenate([arcface, farl_weight * farl]))


def feature_description(feature_mode: str, farl_weight: float) -> str:
    if feature_mode == "arcface":
        return "normalized ArcFace face-recognition vector from aligned face crops"
    if feature_mode == "farl":
        return "normalized FaRL vector"
    return f"normalized ArcFace vector concatenated with {farl_weight:g} * normalized FaRL vector"


def load_face_type_members(
    arcface_csv: Path,
    farl_csv: Path,
    members_csv: Path,
    aliases_csv: Path,
    feature_mode: str,
    farl_weight: float,
    min_image_count: int,
    min_confidence: float,
) -> list[FaceTypeMember]:
    arc_members = {member.member_id: member for member in load_member_vectors(arcface_csv)}
    farl_members: dict[str, MemberVector] = {}
    if feature_mode in {"arcface-farl", "farl"}:
        farl_members = {member.member_id: member for member in load_member_vectors(farl_csv)}
    aliases = load_member_aliases(aliases_csv)
    labels = load_labels(members_csv)

    collapsed: dict[str, MemberVector] = {}
    for member in arc_members.values():
        if feature_mode in {"arcface-farl", "farl"} and member.member_id not in farl_members:
            continue
        if member.image_count < min_image_count or member.confidence < min_confidence:
            continue
        alias_id = aliases.get(member.member_id, member.member_id)
        current = collapsed.get(alias_id)
        if current is None or quality_rank(member) > quality_rank(current):
            collapsed[alias_id] = member

    out: list[FaceTypeMember] = []
    for alias_id, member in collapsed.items():
        feature = build_feature(
            member=member,
            farl_member=farl_members.get(member.member_id),
            feature_mode=feature_mode,
            farl_weight=farl_weight,
        )
        if feature is None:
            continue
        label = labels.get(member.member_id) or f"{member.group_name} {member.member_name}".strip()
        out.append(
            FaceTypeMember(
                member=member,
                alias_id=alias_id,
                label=_clean_label(label),
                feature=feature,
            )
        )
    return out


def choose_cluster_count(
    features: np.ndarray,
    min_k: int,
    max_k: int,
    target_k: int,
    method: str,
) -> tuple[int, list[tuple[int, float, float]]]:
    max_k = min(max_k, max(2, features.shape[0] - 1))
    min_k = min(min_k, max_k)

    scored: list[tuple[int, float, float]] = []
    best_score = -1.0
    best_k = min_k
    sigma = max(2.0, target_k / 3)

    for k in range(min_k, max_k + 1):
        labels = fit_cluster_labels(features, n_clusters=k, method=method)
        silhouette = float(silhouette_score(features, labels, metric="cosine"))
        sizes = np.bincount(labels)
        avg_size = features.shape[0] / k
        balance = min(1.0, (avg_size * 2.5) / float(sizes.max()))
        target_prior = math.exp(-((k - target_k) ** 2) / (2 * sigma * sigma))
        score = max(0.0, silhouette) * balance * target_prior
        scored.append((k, silhouette, score))
        if score > best_score:
            best_score = score
            best_k = k

    return best_k, scored


def fit_cluster_labels(features: np.ndarray, n_clusters: int, method: str) -> np.ndarray:
    if method == "kmeans":
        return KMeans(n_clusters=n_clusters, random_state=42, n_init=30).fit_predict(features)
    return AgglomerativeClustering(
        n_clusters=n_clusters,
        metric="cosine",
        linkage="average",
    ).fit_predict(features)


def cluster_gender(
    members: list[FaceTypeMember],
    min_k: int,
    max_k: int,
    target_k: int,
    method: str,
) -> tuple[np.ndarray, int, float, np.ndarray]:
    features = np.vstack([item.feature for item in members])
    k, _ = choose_cluster_count(
        features,
        min_k=min_k,
        max_k=max_k,
        target_k=target_k,
        method=method,
    )
    labels = fit_cluster_labels(features, n_clusters=k, method=method)
    silhouette = float(silhouette_score(features, labels, metric="cosine"))
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(features)
    return labels, k, silhouette, coords


def nearest_representatives(
    members: list[FaceTypeMember],
    cluster_labels: np.ndarray,
    cluster_id: int,
    top_n: int,
) -> list[tuple[FaceTypeMember, float]]:
    indices = np.where(cluster_labels == cluster_id)[0]
    features = np.vstack([members[i].feature for i in indices])
    centroid = _normalize(features.mean(axis=0))
    scored: list[tuple[FaceTypeMember, float]] = []
    for idx in indices:
        sim = float(members[idx].feature @ centroid)
        scored.append((members[idx], sim))
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:top_n]


def find_crop(member_id: str, crop_root: Path) -> Path | None:
    member_dir = crop_root / member_id
    if not member_dir.exists():
        return None
    for path in sorted(member_dir.iterdir()):
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
            return path
    return None


def make_contact_sheet(
    reps: list[tuple[FaceTypeMember, float]],
    output_path: Path,
    crop_root: Path,
    cols: int = 5,
) -> bool:
    tiles: list[Image.Image] = []
    for item, sim in reps:
        crop_path = find_crop(item.member.member_id, crop_root)
        if crop_path is None:
            continue
        try:
            image = Image.open(crop_path).convert("RGB")
        except Exception:
            continue
        image.thumbnail((128, 128))
        tile = Image.new("RGB", (170, 180), "white")
        tile.paste(image, ((170 - image.width) // 2, 6))
        draw = ImageDraw.Draw(tile)
        draw.text((6, 140), f"{item.member.group_name}", fill="black")
        draw.text((6, 154), f"{item.member.member_name} {sim:.2f}", fill="black")
        tiles.append(tile)

    if not tiles:
        return False

    rows = math.ceil(len(tiles) / cols)
    sheet = Image.new("RGB", (cols * 170, rows * 180), "white")
    for index, tile in enumerate(tiles):
        sheet.paste(tile, ((index % cols) * 170, (index // cols) * 180))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)
    return True


def write_outputs(
    assignments_csv: Path,
    report_md: Path,
    sheet_dir: Path,
    crop_root: Path,
    all_members: dict[str, list[FaceTypeMember]],
    all_labels: dict[str, np.ndarray],
    all_coords: dict[str, np.ndarray],
    all_silhouettes: dict[str, float],
    reps_per_cluster: int,
    method: str,
    feature_text: str,
) -> None:
    assignments_csv.parent.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)

    with assignments_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "face_type_id",
                "gender",
                "cluster_id",
                "member_id",
                "alias_id",
                "label",
                "group_name",
                "member_name",
                "image_count",
                "confidence",
                "x",
                "y",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        for gender in sorted(all_members):
            members = all_members[gender]
            labels = all_labels[gender]
            coords = all_coords[gender]
            cluster_sizes = Counter(labels.tolist())
            ordered_clusters = [
                cid for cid, _ in sorted(cluster_sizes.items(), key=lambda item: (-item[1], item[0]))
            ]
            cluster_to_name = {
                cid: f"{gender}{index + 1:02d}"
                for index, cid in enumerate(ordered_clusters)
            }
            for index, item in enumerate(members):
                writer.writerow(
                    {
                        "face_type_id": cluster_to_name[int(labels[index])],
                        "gender": gender,
                        "cluster_id": int(labels[index]),
                        "member_id": item.member.member_id,
                        "alias_id": item.alias_id,
                        "label": item.label,
                        "group_name": item.member.group_name,
                        "member_name": item.member.member_name,
                        "image_count": item.member.image_count,
                        "confidence": f"{item.member.confidence:.4f}",
                        "x": f"{coords[index, 0]:.4f}",
                        "y": f"{coords[index, 1]:.4f}",
                    }
                )

    lines: list[str] = []
    lines.append("# Face Type Clustering Report")
    lines.append("")
    lines.append("This is an exploratory clustering of idol face embeddings, not an objective taxonomy.")
    lines.append("Known cross-group aliases are collapsed before clustering, and female/male members are clustered separately.")
    lines.append("")
    lines.append("## Method")
    lines.append("")
    lines.append(f"- Feature: {feature_text}")
    lines.append("- Filter: member vectors with at least the requested minimum images/confidence")
    lines.append(f"- Clustering: {method}, cluster count selected near the target size by silhouette/balance")
    lines.append("- Contact sheets: local representative crops nearest to each cluster centroid")
    lines.append("")

    for gender in sorted(all_members):
        members = all_members[gender]
        labels = all_labels[gender]
        cluster_sizes = Counter(labels.tolist())
        ordered_clusters = [
            cid for cid, _ in sorted(cluster_sizes.items(), key=lambda item: (-item[1], item[0]))
        ]
        lines.append(f"## {gender} Clusters")
        lines.append("")
        lines.append(
            f"- Members: {len(members)}"
            f"\n- Clusters: {len(cluster_sizes)}"
            f"\n- Silhouette: {all_silhouettes[gender]:.4f}"
        )
        lines.append("")

        for out_index, cid in enumerate(ordered_clusters, start=1):
            face_type_id = f"{gender}{out_index:02d}"
            reps = nearest_representatives(members, labels, cid, reps_per_cluster)
            sheet_path = sheet_dir / f"{face_type_id}.jpg"
            make_contact_sheet(reps, sheet_path, crop_root=crop_root)

            cluster_members = [members[i] for i in np.where(labels == cid)[0]]
            groups = Counter(item.member.group_name for item in cluster_members).most_common(5)
            avg_conf = float(np.mean([item.member.confidence for item in cluster_members]))
            avg_images = float(np.mean([item.member.image_count for item in cluster_members]))
            rep_text = ", ".join(
                f"{item.member.group_name} {item.member.member_name}" for item, _ in reps[:8]
            )
            group_text = ", ".join(f"{group}({count})" for group, count in groups)
            try:
                sheet_display = sheet_path.relative_to(ROOT_DIR).as_posix()
            except ValueError:
                sheet_display = sheet_path.as_posix()

            lines.append(f"### {face_type_id} ({len(cluster_members)} members)")
            lines.append("")
            lines.append(f"- Avg images/confidence: {avg_images:.1f} / {avg_conf:.3f}")
            lines.append(f"- Top groups: {group_text}")
            lines.append(f"- Representatives: {rep_text}")
            lines.append(f"- Contact sheet: `{sheet_display}`")
            lines.append("")

    report_md.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def run(args: argparse.Namespace) -> None:
    members = load_face_type_members(
        arcface_csv=ROOT_DIR / args.arcface,
        farl_csv=ROOT_DIR / args.farl,
        members_csv=ROOT_DIR / args.members,
        aliases_csv=ROOT_DIR / args.aliases,
        feature_mode=args.feature_mode,
        farl_weight=args.farl_weight,
        min_image_count=args.min_image_count,
        min_confidence=args.min_confidence,
    )

    by_gender: dict[str, list[FaceTypeMember]] = defaultdict(list)
    for item in members:
        if item.member.gender in {"F", "M"}:
            by_gender[item.member.gender].append(item)

    all_labels: dict[str, np.ndarray] = {}
    all_coords: dict[str, np.ndarray] = {}
    all_silhouettes: dict[str, float] = {}

    for gender, items in sorted(by_gender.items()):
        labels, k, silhouette, coords = cluster_gender(
            items,
            min_k=args.min_k,
            max_k=args.max_k,
            target_k=args.target_k,
            method=args.method,
        )
        all_labels[gender] = labels
        all_coords[gender] = coords
        all_silhouettes[gender] = silhouette
        print(f"{gender}: members={len(items)} clusters={k} silhouette={silhouette:.4f}")

    write_outputs(
        assignments_csv=ROOT_DIR / args.assignments_csv,
        report_md=ROOT_DIR / args.report_md,
        sheet_dir=ROOT_DIR / args.sheet_dir,
        crop_root=ROOT_DIR / args.crop_root,
        all_members=dict(by_gender),
        all_labels=all_labels,
        all_coords=all_coords,
        all_silhouettes=all_silhouettes,
        reps_per_cluster=args.representatives,
        method=args.method,
        feature_text=feature_description(args.feature_mode, args.farl_weight),
    )
    print(f"Wrote {args.assignments_csv}")
    print(f"Wrote {args.report_md}")
    print(f"Wrote contact sheets to {args.sheet_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explore face-type clusters from member vectors.")
    parser.add_argument("--arcface", default="data/member_vectors.csv")
    parser.add_argument("--farl", default="data/member_vectors_farl.csv")
    parser.add_argument("--members", default="data/members.csv")
    parser.add_argument("--aliases", default="data/member_aliases.csv")
    parser.add_argument("--assignments-csv", default="data/face_type_clusters.csv")
    parser.add_argument("--report-md", default="docs/face_type_clusters.md")
    parser.add_argument("--sheet-dir", default=".cache/face_type_clusters")
    parser.add_argument("--crop-root", default="data/face_crops")
    parser.add_argument(
        "--feature-mode",
        choices=["arcface", "arcface-farl", "farl"],
        default="arcface",
        help="arcface is face-only and ignores styling/context; arcface-farl matches the older fused exploratory setup.",
    )
    parser.add_argument("--farl-weight", type=float, default=0.5)
    parser.add_argument("--min-image-count", type=int, default=5)
    parser.add_argument("--min-confidence", type=float, default=0.35)
    parser.add_argument("--min-k", type=int, default=7)
    parser.add_argument("--max-k", type=int, default=14)
    parser.add_argument("--target-k", type=int, default=10)
    parser.add_argument("--representatives", type=int, default=10)
    parser.add_argument("--method", choices=["kmeans", "agglomerative"], default="kmeans")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
