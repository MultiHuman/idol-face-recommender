from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ImageEmbedding:
    member_id: str
    group_name: str
    member_name: str
    image_path: str
    vector: np.ndarray


def parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y"}


def parse_vector(vector_json: str) -> np.ndarray:
    values = json.loads(vector_json)
    vector = np.asarray(values, dtype=np.float32)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError("Embedding vector must be a non-empty 1D array.")
    return vector


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    safe_norms = np.clip(norms, 1e-12, None)
    return matrix / safe_norms


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError("Vector norm is zero.")
    return vector / norm


def load_embeddings(csv_path: str | Path) -> dict[str, list[ImageEmbedding]]:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Input CSV does not exist: {path}")

    grouped: dict[str, list[ImageEmbedding]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if not parse_bool(row.get("is_valid_face") or "false"):
                continue

            member_id = (row.get("member_id") or "").strip()
            if not member_id:
                continue

            grouped.setdefault(member_id, []).append(
                ImageEmbedding(
                    member_id=member_id,
                    group_name=(row.get("group_name") or "").strip(),
                    member_name=(row.get("member_name") or "").strip(),
                    image_path=(row.get("image_path") or "").strip(),
                    vector=parse_vector(row.get("vector_json") or "[]"),
                )
            )
    return grouped


def compute_member_summary(items: list[ImageEmbedding]) -> dict[str, str | int | float]:
    matrix = np.vstack([item.vector for item in items])
    normalized = normalize_rows(matrix)
    pairwise = normalized @ normalized.T
    upper_indices = np.triu_indices(len(items), k=1)
    pairwise_values = pairwise[upper_indices]

    centroid = normalize_vector(matrix.mean(axis=0))
    centroid_sims = normalized @ centroid
    worst_index = int(np.argmin(centroid_sims))

    return {
        "member_id": items[0].member_id,
        "group_name": items[0].group_name,
        "member_name": items[0].member_name,
        "image_count": len(items),
        "pair_count": int(pairwise_values.size),
        "mean_pairwise_cosine": float(pairwise_values.mean()),
        "median_pairwise_cosine": float(np.median(pairwise_values)),
        "min_pairwise_cosine": float(pairwise_values.min()),
        "max_pairwise_cosine": float(pairwise_values.max()),
        "mean_centroid_cosine": float(centroid_sims.mean()),
        "min_centroid_cosine": float(centroid_sims.min()),
        "max_centroid_cosine": float(centroid_sims.max()),
        "worst_image_path": items[worst_index].image_path,
    }


def leave_one_out_accuracy(grouped: dict[str, list[ImageEmbedding]]) -> dict[str, float | int]:
    eligible = {member_id: items for member_id, items in grouped.items() if len(items) >= 2}
    if not eligible:
        return {
            "eligible_members": 0,
            "eligible_images": 0,
            "top1_accuracy": 0.0,
            "mean_own_similarity": 0.0,
            "mean_best_other_similarity": 0.0,
            "mean_margin": 0.0,
        }

    normalized_by_member = {
        member_id: normalize_rows(np.vstack([item.vector for item in items]))
        for member_id, items in eligible.items()
    }
    full_centroids = {
        member_id: normalize_vector(np.vstack([item.vector for item in items]).mean(axis=0))
        for member_id, items in eligible.items()
    }

    correct = 0
    total = 0
    own_similarities: list[float] = []
    best_other_similarities: list[float] = []
    margins: list[float] = []

    for member_id, items in eligible.items():
        member_vectors = normalized_by_member[member_id]
        for index, _item in enumerate(items):
            mask = np.ones(len(items), dtype=bool)
            mask[index] = False
            own_centroid = normalize_vector(member_vectors[mask].mean(axis=0))
            probe = member_vectors[index]

            similarities: dict[str, float] = {}
            similarities[member_id] = float(probe @ own_centroid)
            for other_member_id, centroid in full_centroids.items():
                if other_member_id == member_id:
                    continue
                similarities[other_member_id] = float(probe @ centroid)

            predicted_member = max(similarities, key=similarities.get)
            best_other_similarity = max(
                similarity
                for candidate_member_id, similarity in similarities.items()
                if candidate_member_id != member_id
            )
            own_similarity = similarities[member_id]

            total += 1
            if predicted_member == member_id:
                correct += 1

            own_similarities.append(own_similarity)
            best_other_similarities.append(best_other_similarity)
            margins.append(own_similarity - best_other_similarity)

    return {
        "eligible_members": len(eligible),
        "eligible_images": total,
        "top1_accuracy": correct / total if total else 0.0,
        "mean_own_similarity": float(np.mean(own_similarities)) if own_similarities else 0.0,
        "mean_best_other_similarity": float(np.mean(best_other_similarities)) if best_other_similarities else 0.0,
        "mean_margin": float(np.mean(margins)) if margins else 0.0,
    }


def write_summary_csv(output_path: str | Path, rows: list[dict[str, str | int | float]]) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "member_id",
        "group_name",
        "member_name",
        "image_count",
        "pair_count",
        "mean_pairwise_cosine",
        "median_pairwise_cosine",
        "min_pairwise_cosine",
        "max_pairwise_cosine",
        "mean_centroid_cosine",
        "min_centroid_cosine",
        "max_centroid_cosine",
        "worst_image_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze how consistent same-member face embeddings are.")
    parser.add_argument(
        "--input",
        default="data/image_embeddings.csv",
        help="Path to the image-level embeddings CSV file.",
    )
    parser.add_argument(
        "--min-images",
        type=int,
        default=2,
        help="Minimum number of valid images required for a member to be included in the report.",
    )
    parser.add_argument(
        "--summary-output",
        default="data/embedding_similarity_report.csv",
        help="Path to the per-member summary CSV file.",
    )
    parser.add_argument(
        "--show-bottom",
        type=int,
        default=10,
        help="How many low-similarity members to print.",
    )
    args = parser.parse_args()

    grouped = load_embeddings(args.input)
    eligible = {member_id: items for member_id, items in grouped.items() if len(items) >= args.min_images}
    if not eligible:
        raise SystemExit(f"No members with at least {args.min_images} valid images were found.")

    summary_rows = [compute_member_summary(items) for items in eligible.values()]
    summary_rows.sort(key=lambda row: float(row["mean_pairwise_cosine"]))
    write_summary_csv(args.summary_output, summary_rows)

    pairwise_scores = [float(row["mean_pairwise_cosine"]) for row in summary_rows]
    centroid_scores = [float(row["mean_centroid_cosine"]) for row in summary_rows]
    leave_one_out = leave_one_out_accuracy(eligible)

    print(
        f"Members analyzed: {len(summary_rows)} | "
        f"Overall mean pairwise cosine: {np.mean(pairwise_scores):.4f} | "
        f"Median pairwise cosine: {np.median(pairwise_scores):.4f} | "
        f"Overall mean centroid cosine: {np.mean(centroid_scores):.4f}"
    )
    print(
        "Leave-one-out member match | "
        f"Eligible members: {int(leave_one_out['eligible_members'])} | "
        f"Eligible images: {int(leave_one_out['eligible_images'])} | "
        f"Top-1 accuracy: {float(leave_one_out['top1_accuracy']):.4f} | "
        f"Mean own similarity: {float(leave_one_out['mean_own_similarity']):.4f} | "
        f"Mean best other similarity: {float(leave_one_out['mean_best_other_similarity']):.4f} | "
        f"Mean margin: {float(leave_one_out['mean_margin']):.4f}"
    )
    print(f"Lowest-consistency members (bottom {min(args.show_bottom, len(summary_rows))}):")
    for row in summary_rows[: args.show_bottom]:
        print(
            f"- {row['member_id']} | images={row['image_count']} | "
            f"mean_pairwise={float(row['mean_pairwise_cosine']):.4f} | "
            f"min_pairwise={float(row['min_pairwise_cosine']):.4f} | "
            f"worst={row['worst_image_path']}"
        )
    print(f"Summary CSV: {args.summary_output}")


if __name__ == "__main__":
    main()
