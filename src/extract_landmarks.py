"""얼굴 5-point 키포인트 + bbox 기반 기하학 특징 추출."""
from __future__ import annotations

import argparse
import csv
import os
import warnings
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT_DIR / ".cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


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
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
warnings.filterwarnings("ignore", category=FutureWarning)

import cv2
import numpy as np
from insightface.app import FaceAnalysis


def unicode_imread(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


FEATURE_NAMES = [
    "face_aspect",          # bbox 가로/세로 (>1이면 옆으로 넓은 얼굴)
    "eye_spacing",          # 눈 사이 거리 / bbox 너비
    "eye_position_y",       # 눈 중심 y / bbox 세로 (0=이마끝, 1=턱끝) — 이마 비율 반영
    "nose_length",          # 코끝 y - 눈 중심 y / bbox 세로 (중안면 길이)
    "philtrum",             # 입 중심 y - 코끝 y / bbox 세로 (인중 길이)
    "chin_length",          # bbox 하단 - 입 중심 y / bbox 세로 (하관 길이)
    "mouth_width",          # 입 가로 / bbox 너비
    "mouth_to_eye_ratio",   # 입 너비 / 눈 간격 (입이 눈 대비 얼마나 큰지)
    "nose_offset_x",        # 코끝 x - 눈 중심 x / bbox 너비 (얼굴 정면 대칭성)
    "lower_face_ratio",     # 하관 / 중안면 (턱이 얼마나 긴 편인지)
]


def compute_features_from_kps(bbox: np.ndarray, kps: np.ndarray) -> dict[str, float] | None:
    """
    kps: [left_eye, right_eye, nose, left_mouth, right_mouth] — 원본 이미지 좌표
    bbox: [x0, y0, x1, y1]
    """
    if bbox is None or kps is None or kps.shape != (5, 2):
        return None

    left_eye = kps[0].astype(np.float32)
    right_eye = kps[1].astype(np.float32)
    nose = kps[2].astype(np.float32)
    left_mouth = kps[3].astype(np.float32)
    right_mouth = kps[4].astype(np.float32)

    # 1) 눈 기울기로 회전 각도 계산
    eye_vec = right_eye - left_eye
    eye_dist = float(np.linalg.norm(eye_vec))
    if eye_dist < 1e-6:
        return None
    angle = float(np.arctan2(eye_vec[1], eye_vec[0]))  # radians

    # 2) 얼굴 중심 (눈 중점) 기준으로 모든 점과 bbox 코너 회전
    eye_mid = (left_eye + right_eye) / 2.0
    cos_a = float(np.cos(-angle))
    sin_a = float(np.sin(-angle))
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)

    def rot(pt: np.ndarray) -> np.ndarray:
        return R @ (pt - eye_mid)

    le = rot(left_eye)
    re = rot(right_eye)
    no = rot(nose)
    lm = rot(left_mouth)
    rm = rot(right_mouth)

    # bbox 4개 코너 회전 후 새 bbox 계산
    corners = np.array([
        [bbox[0], bbox[1]],
        [bbox[2], bbox[1]],
        [bbox[2], bbox[3]],
        [bbox[0], bbox[3]],
    ], dtype=np.float32)
    rotated_corners = np.array([rot(c) for c in corners])
    bx0, by0 = rotated_corners[:, 0].min(), rotated_corners[:, 1].min()
    bx1, by1 = rotated_corners[:, 0].max(), rotated_corners[:, 1].max()
    face_w = float(bx1 - bx0)
    face_h = float(by1 - by0)
    if face_w < 1e-6 or face_h < 1e-6:
        return None

    # 3) 회전된 좌표에서 수직/수평 위치 추출
    eye_y = float((le[1] + re[1]) / 2.0)
    mouth_mid = (lm + rm) / 2.0
    mouth_y = float(mouth_mid[1])
    nose_y = float(no[1])
    eye_x = float((le[0] + re[0]) / 2.0)
    nose_x = float(no[0])
    mouth_w = float(np.linalg.norm(rm - lm))

    # bbox 기준 상대 위치 (bbox 상단=0, 하단=1)
    eye_y_rel = (eye_y - by0) / face_h
    nose_y_rel = (nose_y - by0) / face_h
    mouth_y_rel = (mouth_mid[1] - by0) / face_h
    chin_y_rel = 1.0  # by definition

    mid_face = max(nose_y_rel - eye_y_rel, 1e-6)  # 중안면 길이 (비율)
    lower_face = max(chin_y_rel - mouth_y_rel, 1e-6)  # 하관 길이 (비율)

    features = {
        "face_aspect": face_w / face_h,
        "eye_spacing": float(np.linalg.norm(re - le)) / face_w,
        "eye_position_y": eye_y_rel,
        "nose_length": mid_face,
        "philtrum": max(mouth_y_rel - nose_y_rel, 0.0),
        "chin_length": lower_face,
        "mouth_width": mouth_w / face_w,
        "mouth_to_eye_ratio": mouth_w / max(np.linalg.norm(re - le), 1e-6),
        "nose_offset_x": (nose_x - eye_x) / face_w,
        "lower_face_ratio": lower_face / mid_face,
    }
    return features


def init_analyzer(ctx_id: int) -> FaceAnalysis:
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if ctx_id >= 0 else ["CPUExecutionProvider"]
    app = FaceAnalysis(
        name="buffalo_l",
        root=str(CACHE_DIR / "insightface"),
        allowed_modules=["detection"],
        providers=providers,
    )
    app.prepare(ctx_id=ctx_id, det_size=(640, 640))
    return app


def load_valid_faces(embeddings_csv: Path) -> list[dict[str, str]]:
    with embeddings_csv.open(encoding="utf-8-sig", newline="") as f:
        return [row for row in csv.DictReader(f) if row.get("is_valid_face") == "true"]


def run(embeddings_csv: Path, output_csv: Path, ctx_id: int) -> None:
    rows = load_valid_faces(embeddings_csv)
    print(f"Valid face rows: {len(rows)}")

    analyzer = init_analyzer(ctx_id=ctx_id)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["member_id", "image_path"] + FEATURE_NAMES
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        processed = 0
        ok = 0
        for row in rows:
            img_path = ROOT_DIR / row["image_path"]
            if not img_path.exists():
                continue
            img = unicode_imread(img_path)
            if img is None:
                continue

            faces = analyzer.get(img)
            if not faces:
                processed += 1
                continue
            face = faces[0]
            bbox = np.asarray(face.bbox, dtype=np.float32)
            kps = np.asarray(face.kps, dtype=np.float32)
            features = compute_features_from_kps(bbox, kps)
            if features is None:
                processed += 1
                continue

            out = {"member_id": row["member_id"], "image_path": row["image_path"]}
            out.update({k: f"{v:.5f}" for k, v in features.items()})
            writer.writerow(out)
            ok += 1
            processed += 1
            if processed % 1000 == 0:
                print(f"  {processed}/{len(rows)} processed, {ok} features extracted")

    print(f"Done. {ok}/{processed} features extracted → {output_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract 5-point keypoint based geometric features.")
    parser.add_argument("--embeddings", default="data/image_embeddings.csv")
    parser.add_argument("--output", default="data/image_landmarks.csv")
    parser.add_argument("--ctx-id", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(
        embeddings_csv=ROOT_DIR / args.embeddings,
        output_csv=ROOT_DIR / args.output,
        ctx_id=args.ctx_id,
    )


if __name__ == "__main__":
    main()
