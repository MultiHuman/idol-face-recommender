"""FaRL (Facial Representation Learning) 기반 얼굴 임베딩 추출.

FaRL은 CLIP ViT-B/16 구조에 LAION-FACE 2천만 장의 얼굴-텍스트 페어로 학습된 모델.
ArcFace가 identity 구분에 최적화된 것과 달리, FaRL은 "이 얼굴이 어떻게 묘사되는가"를
학습해서 시각적/스타일적 유사성이 임베딩에 담김.

사용법:
    # 1. CLIP 설치: pip install git+https://github.com/openai/CLIP.git
    # 2. 가중치 다운: .cache/farl/FaRL-Base-Patch16-LAIONFace20M-ep64.pth
    # 3. 실행: python -m src.extract_farl
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

try:
    import clip
except ImportError as exc:
    raise SystemExit(
        "OpenAI CLIP이 설치되지 않았어. 설치 명령어:\n"
        "  pip install git+https://github.com/openai/CLIP.git"
    ) from exc


ROOT_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT_DIR / ".cache"
FARL_WEIGHTS = CACHE_DIR / "farl" / "FaRL-Base-Patch16-LAIONFace20M-ep64.pth"


def load_farl_model(device: str = "cuda") -> tuple[torch.nn.Module, callable]:
    """CLIP ViT-B/16 구조에 FaRL 가중치를 덮어씌워서 로딩."""
    if not FARL_WEIGHTS.exists():
        raise SystemExit(
            f"FaRL 가중치가 없어: {FARL_WEIGHTS}\n"
            f"다운로드: https://github.com/FacePerceiver/FaRL/releases/download/"
            f"pretrained_weights/FaRL-Base-Patch16-LAIONFace20M-ep64.pth"
        )

    # OpenAI CLIP ViT-B/16을 base로 로딩 (네트워크 구조만 사용)
    print(f"Loading CLIP ViT-B/16 base model on {device}...")
    model, preprocess = clip.load("ViT-B/16", device=device, jit=False)

    print(f"Loading FaRL weights from {FARL_WEIGHTS.name}...")
    farl_state = torch.load(str(FARL_WEIGHTS), map_location=device, weights_only=False)
    # FaRL 체크포인트는 여러 포맷이 있을 수 있음
    if isinstance(farl_state, dict):
        if "state_dict" in farl_state:
            farl_state = farl_state["state_dict"]
        elif "model" in farl_state:
            farl_state = farl_state["model"]

    # CLIP의 state_dict와 key가 다를 수 있으니 matching되는 것만 로드
    current_state = model.state_dict()
    loaded_keys = 0
    skipped_keys = 0
    for k, v in farl_state.items():
        if k in current_state and current_state[k].shape == v.shape:
            current_state[k] = v
            loaded_keys += 1
        else:
            skipped_keys += 1
    model.load_state_dict(current_state)
    print(f"Loaded {loaded_keys} matching keys, skipped {skipped_keys} non-matching.")

    model.eval()
    return model, preprocess


def encode_image_pil(model: torch.nn.Module, preprocess, pil_img: Image.Image, device: str) -> np.ndarray:
    """PIL 이미지 → FaRL 512D 임베딩 (L2 정규화 적용)."""
    with torch.no_grad():
        img_tensor = preprocess(pil_img).unsqueeze(0).to(device)
        feat = model.encode_image(img_tensor)
        feat = feat / feat.norm(dim=-1, keepdim=True)  # L2 normalize
    return feat.squeeze(0).cpu().float().numpy()


def load_valid_face_rows(embeddings_csv: Path) -> list[dict[str, str]]:
    with embeddings_csv.open(encoding="utf-8-sig", newline="") as f:
        return [row for row in csv.DictReader(f) if row.get("is_valid_face") == "true"]


def run(
    embeddings_csv: Path,
    output_csv: Path,
    image_root: Path,
    use_crops: bool,
    device: str,
    batch_size: int,
) -> None:
    rows = load_valid_face_rows(embeddings_csv)
    print(f"Valid face rows: {len(rows)}")

    model, preprocess = load_farl_model(device=device)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["member_id", "image_path", "farl_vector_json"])
        writer.writeheader()

        processed = 0
        ok = 0
        batch_pils: list[Image.Image] = []
        batch_rows: list[dict[str, str]] = []

        def flush_batch() -> int:
            nonlocal ok
            if not batch_pils:
                return 0
            with torch.no_grad():
                tensors = torch.stack([preprocess(img) for img in batch_pils]).to(device)
                feats = model.encode_image(tensors)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            for row, feat in zip(batch_rows, feats.cpu().float().numpy()):
                writer.writerow({
                    "member_id": row["member_id"],
                    "image_path": row["image_path"],
                    "farl_vector_json": json.dumps(feat.tolist(), separators=(",", ":")),
                })
                ok += 1
            n = len(batch_pils)
            batch_pils.clear()
            batch_rows.clear()
            return n

        for row in rows:
            # use_crops=True면 face_crops 경로 시도, 아니면 raw image
            if use_crops and row.get("crop_path"):
                img_path = ROOT_DIR / row["crop_path"]
                if not img_path.exists():
                    img_path = ROOT_DIR / row["image_path"]
            else:
                img_path = ROOT_DIR / row["image_path"]

            if not img_path.exists():
                processed += 1
                continue

            try:
                pil = Image.open(img_path).convert("RGB")
            except Exception:
                processed += 1
                continue

            batch_pils.append(pil)
            batch_rows.append(row)
            processed += 1

            if len(batch_pils) >= batch_size:
                flush_batch()
                if processed % 500 == 0 or processed == len(rows):
                    print(f"  {processed}/{len(rows)} processed, {ok} embeddings extracted")

        flush_batch()
        print(f"Done. {ok}/{processed} FaRL embeddings extracted → {output_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract FaRL embeddings from valid face images.")
    parser.add_argument("--embeddings", default="data/image_embeddings.csv")
    parser.add_argument("--output", default="data/image_farl.csv")
    parser.add_argument("--image-root", default=".", help="경로 기준 루트 (기본 프로젝트 루트)")
    parser.add_argument("--use-crops", action="store_true", help="face_crops/ 폴더 우선 사용 (기본: raw image)")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA unavailable, falling back to CPU.")
        device = "cpu"
    run(
        embeddings_csv=ROOT_DIR / args.embeddings,
        output_csv=ROOT_DIR / args.output,
        image_root=ROOT_DIR / args.image_root,
        use_crops=args.use_crops,
        device=device,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
