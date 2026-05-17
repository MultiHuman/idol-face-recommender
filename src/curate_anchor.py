"""kprofiles 에서 가져온 앵커 이미지를 기준으로 기존 사진을 필터링.

흐름:
1) .cache/anchors/{member_id}.jpg 들에 FaRL 임베딩 계산 (한 번만)
2) image_farl.csv 의 각 사진과 해당 멤버 앵커 간 코사인 유사도 계산
3) 임계값 미만 사진 삭제 (원본, 크롭, image_farl.csv 행 제거, image_embeddings.csv 는 is_valid_face=false)

앵커가 없는 멤버는 건너뜀 (영향 없음).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import warnings
from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT_DIR / ".cache"


def _add_cuda_dll_directories() -> None:
    if os.name != "nt":
        return
    import sys
    candidates: list[Path] = []
    for site_dir in sys.path:
        nvidia_dir = Path(site_dir) / "nvidia"
        if nvidia_dir.is_dir():
            for sub in nvidia_dir.iterdir():
                bin_dir = sub / "bin"
                if bin_dir.is_dir():
                    candidates.append(bin_dir)
    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path:
        candidates.append(Path(cuda_path) / "bin")
    base = Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA")
    if base.exists():
        for v in sorted((p for p in base.glob("v*") if p.is_dir()), key=lambda p: p.name, reverse=True):
            candidates.append(v / "bin")
    seen: set[Path] = set()
    new_paths: list[str] = []
    for p in candidates:
        if p in seen or not p.exists():
            continue
        seen.add(p)
        try:
            os.add_dll_directory(str(p))
        except (OSError, AttributeError):
            pass
        new_paths.append(str(p))
    if new_paths:
        os.environ["PATH"] = os.pathsep.join(new_paths) + os.pathsep + os.environ.get("PATH", "")


_add_cuda_dll_directories()
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np


@dataclass
class ImageRecord:
    member_id: str
    image_path: str
    vector: np.ndarray


def compute_anchor_embeddings(
    anchor_dir: Path,
    output_csv: Path,
    device: str = "cuda",
) -> dict[str, np.ndarray]:
    """앵커 이미지들에 FaRL 임베딩 계산 후 CSV 저장 & dict 리턴."""
    # 이미 캐시돼 있으면 로드
    if output_csv.exists():
        print(f"Loading cached anchor embeddings from {output_csv}")
        result: dict[str, np.ndarray] = {}
        with output_csv.open(encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                mid = row["member_id"]
                vec = np.asarray(json.loads(row["farl_vector_json"]), dtype=np.float32)
                result[mid] = vec
        return result

    import torch
    import clip
    from PIL import Image

    farl_weights = CACHE_DIR / "farl" / "FaRL-Base-Patch16-LAIONFace20M-ep64.pth"
    if not farl_weights.exists():
        raise SystemExit(f"FaRL weights not found: {farl_weights}")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    print(f"Loading CLIP ViT-B/16 + FaRL weights on {device}...")
    model, preprocess = clip.load("ViT-B/16", device=device, jit=False)
    farl_state = torch.load(str(farl_weights), map_location=device, weights_only=False)
    if isinstance(farl_state, dict):
        if "state_dict" in farl_state:
            farl_state = farl_state["state_dict"]
        elif "model" in farl_state:
            farl_state = farl_state["model"]
    current = model.state_dict()
    loaded = 0
    for k, v in farl_state.items():
        if k in current and current[k].shape == v.shape:
            current[k] = v
            loaded += 1
    model.load_state_dict(current)
    print(f"FaRL loaded ({loaded} keys)")
    model.eval()

    anchor_files = sorted(anchor_dir.glob("*.jpg"))
    print(f"Computing embeddings for {len(anchor_files)} anchor images...")

    result = {}
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["member_id", "anchor_path", "farl_vector_json"])
        writer.writeheader()
        ok = 0
        fail = 0
        for path in anchor_files:
            mid = path.stem
            try:
                pil = Image.open(path).convert("RGB")
                with torch.no_grad():
                    tensor = preprocess(pil).unsqueeze(0).to(device)
                    feat = model.encode_image(tensor)
                    feat = feat / feat.norm(dim=-1, keepdim=True)
                vec = feat.squeeze(0).cpu().float().numpy()
            except Exception as exc:
                print(f"  [fail] {mid}: {exc}")
                fail += 1
                continue
            result[mid] = vec
            writer.writerow({
                "member_id": mid,
                "anchor_path": str(path.relative_to(ROOT_DIR)),
                "farl_vector_json": json.dumps(vec.tolist(), separators=(",", ":")),
            })
            ok += 1
            if ok % 100 == 0:
                print(f"  {ok}/{len(anchor_files)}")
        print(f"Done: {ok} OK, {fail} failed")
    return result


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


def load_arcface_image_records(embeddings_csv: Path) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    with embeddings_csv.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("is_valid_face") != "true":
                continue
            mid = (row.get("member_id") or "").strip()
            path = (row.get("image_path") or "").strip()
            vec_json = (row.get("vector_json") or "").strip()
            if not mid or not path or not vec_json:
                continue
            vec = np.asarray(json.loads(vec_json), dtype=np.float32)
            # L2 normalize (ArcFace 벡터가 정규화 안 돼있을 수 있음)
            norm = float(np.linalg.norm(vec))
            if norm > 1e-6:
                vec = vec / norm
            records.append(ImageRecord(member_id=mid, image_path=path, vector=vec))
    return records


def load_anchor_arcface(anchor_csv: Path) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    if not anchor_csv.exists():
        return result
    with anchor_csv.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            mid = row.get("member_id", "").strip()
            vec_json = row.get("vector_json", "").strip()
            if not mid or not vec_json:
                continue
            result[mid] = np.asarray(json.loads(vec_json), dtype=np.float32)
    return result


def mark_below_threshold(
    records: list[ImageRecord],
    anchors: dict[str, np.ndarray],
    min_similarity: float,
) -> list[tuple[ImageRecord, float]]:
    marked: list[tuple[ImageRecord, float]] = []
    for rec in records:
        anchor = anchors.get(rec.member_id)
        if anchor is None:
            continue  # 앵커 없는 멤버는 건너뜀
        sim = float(np.dot(rec.vector, anchor))
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


def _quarantine_or_delete(path: Path, quarantine_root: Path | None) -> bool:
    if not path.exists():
        return False
    if quarantine_root is None:
        path.unlink()
        return True

    rel = path.resolve().relative_to(ROOT_DIR.resolve())
    destination = quarantine_root / rel
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination = destination.with_name(f"{destination.stem}_{os.getpid()}{destination.suffix}")
    shutil.move(str(path), str(destination))
    return True


def delete_files(
    marked: list[tuple[ImageRecord, float]],
    raw_root: Path,
    crop_root: Path,
    quarantine_root: Path | None = None,
) -> int:
    removed = 0
    for rec, _ in marked:
        raw_path = ROOT_DIR / rec.image_path
        try:
            if _quarantine_or_delete(raw_path, quarantine_root):
                removed += 1
                prune_empty_directories(raw_path.parent, raw_root)
        except (OSError, ValueError):
            pass
        crop_dir = crop_root / rec.member_id
        if crop_dir.exists():
            for cand in crop_dir.glob(f"{raw_path.stem}.*"):
                try:
                    _quarantine_or_delete(cand, quarantine_root)
                except OSError:
                    pass
            prune_empty_directories(crop_dir, crop_root)
    return removed


def update_embeddings_csv(embeddings_csv: Path, removed_paths: set[str]) -> int:
    if not embeddings_csv.exists() or not removed_paths:
        return 0
    with embeddings_csv.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    updated = 0
    for row in rows:
        row.pop(None, None)
        if (row.get("image_path") or "").strip() in removed_paths:
            row["is_valid_face"] = "false"
            row["vector_json"] = ""
            updated += 1
    with embeddings_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return updated


def update_farl_csv(farl_csv: Path, removed_paths: set[str]) -> int:
    if not farl_csv.exists() or not removed_paths:
        return 0
    with farl_csv.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    for r in rows:
        r.pop(None, None)
    kept = [r for r in rows if (r.get("image_path") or "").strip() not in removed_paths]
    with farl_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(kept)
    return len(rows) - len(kept)


def run(
    anchor_dir: Path,
    anchor_csv: Path,
    image_farl_csv: Path,
    embeddings_csv: Path,
    raw_root: Path,
    crop_root: Path,
    min_similarity: float,
    dry_run: bool,
    mode: str = "farl",
    anchor_arcface_csv: Path | None = None,
    member_ids: set[str] | None = None,
    quarantine_root: Path | None = None,
) -> None:
    if mode == "arcface":
        if anchor_arcface_csv is None:
            raise SystemExit("arcface mode requires --anchor-arcface-csv path")
        anchors = load_anchor_arcface(anchor_arcface_csv)
        if not anchors:
            raise SystemExit(f"Anchor ArcFace CSV empty or missing: {anchor_arcface_csv}")
        print(f"Loaded {len(anchors)} ArcFace anchors.")
        records = load_arcface_image_records(embeddings_csv)
        print(f"Loaded {len(records)} image ArcFace records.")
    else:
        anchors = compute_anchor_embeddings(anchor_dir, anchor_csv)
        print(f"Loaded {len(anchors)} FaRL anchors.")
        records = load_image_records(image_farl_csv)
        print(f"Loaded {len(records)} image FaRL records.")

    if member_ids:
        records = [record for record in records if record.member_id in member_ids]
        anchors = {mid: vector for mid, vector in anchors.items() if mid in member_ids}
        print(f"Filtered to {len(records)} image records for {len(member_ids)} requested members.")

    marked = mark_below_threshold(records, anchors, min_similarity)
    print(f"Marked {len(marked)} images for removal (anchor cosine < {min_similarity}, mode={mode}).")

    if dry_run:
        from collections import defaultdict
        per_member: dict[str, list[float]] = defaultdict(list)
        for rec, sim in marked:
            per_member[rec.member_id].append(sim)
        print("\nTop members with most removed:")
        for mid, lst in sorted(per_member.items(), key=lambda x: -len(x[1]))[:20]:
            print(f"  {mid}: {len(lst)} removed (worst: {min(lst):.3f})")
        # 분포
        sims = [s for _, s in marked]
        if sims:
            import statistics
            print(f"\nRemoved similarity: min={min(sims):.3f}, max={max(sims):.3f}, median={statistics.median(sims):.3f}")
        print("\n(dry-run. 파일 삭제하지 않음.)")
        return

    removed_paths = {rec.image_path for rec, _ in marked}
    removed = delete_files(marked, raw_root, crop_root, quarantine_root=quarantine_root)
    action = "Quarantined" if quarantine_root is not None else "Deleted"
    print(f"{action} {removed} raw image files.")
    emb_updated = update_embeddings_csv(embeddings_csv, removed_paths)
    print(f"Updated {emb_updated} rows in {embeddings_csv.name}.")
    farl_removed = update_farl_csv(image_farl_csv, removed_paths)
    print(f"Removed {farl_removed} rows from {image_farl_csv.name}.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Curate images by kprofiles anchor similarity.")
    parser.add_argument("--anchor-dir", default=".cache/anchors")
    parser.add_argument("--anchor-csv", default="data/member_anchors_farl.csv")
    parser.add_argument("--image-farl", default="data/image_farl.csv")
    parser.add_argument("--embeddings", default="data/image_embeddings.csv")
    parser.add_argument("--raw-root", default="data/raw_images")
    parser.add_argument("--crop-root", default="data/face_crops")
    parser.add_argument("--min-similarity", type=float, default=0.75)
    parser.add_argument("--mode", choices=["farl", "arcface"], default="farl")
    parser.add_argument("--anchor-arcface-csv", default="data/member_anchors_arcface.csv")
    parser.add_argument("--member-ids", nargs="*", help="Only curate these member IDs.")
    parser.add_argument("--quarantine-dir", default=None, help="Move removed files here instead of deleting them.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(
        anchor_dir=ROOT_DIR / args.anchor_dir,
        anchor_csv=ROOT_DIR / args.anchor_csv,
        image_farl_csv=ROOT_DIR / args.image_farl,
        embeddings_csv=ROOT_DIR / args.embeddings,
        raw_root=ROOT_DIR / args.raw_root,
        crop_root=ROOT_DIR / args.crop_root,
        min_similarity=args.min_similarity,
        dry_run=args.dry_run,
        mode=args.mode,
        anchor_arcface_csv=ROOT_DIR / args.anchor_arcface_csv,
        member_ids={mid.strip() for mid in args.member_ids if mid.strip()} if args.member_ids else None,
        quarantine_root=ROOT_DIR / args.quarantine_dir if args.quarantine_dir else None,
    )


if __name__ == "__main__":
    main()
