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
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# CUDA DLL 경로 자동 추가 (GPU 가속을 위해)
def _add_cuda_dll_directories() -> None:
    if os.name != "nt":
        return
    candidates: list[Path] = []

    # 1) pip로 설치된 nvidia-* 패키지 (가장 우선 — onnxruntime과 버전 매칭이 정확함)
    import sys
    for site_dir in sys.path:
        nvidia_dir = Path(site_dir) / "nvidia"
        if nvidia_dir.is_dir():
            for sub in nvidia_dir.iterdir():
                bin_dir = sub / "bin"
                if bin_dir.is_dir():
                    candidates.append(bin_dir)

    # 2) 환경변수 CUDA_PATH (NVIDIA toolkit 설치 시 자동 설정됨)
    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path:
        candidates.append(Path(cuda_path) / "bin")

    # 3) 일반적인 NVIDIA 설치 경로 — 가장 높은 v* 버전을 우선
    base = Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA")
    if base.exists():
        versions = sorted(
            (p for p in base.glob("v*") if p.is_dir()),
            key=lambda p: p.name,
            reverse=True,
        )
        candidates.extend(v / "bin" for v in versions)

    seen: set[Path] = set()
    new_path_parts: list[str] = []
    for path in candidates:
        if path in seen or not path.exists():
            continue
        seen.add(path)
        try:
            os.add_dll_directory(str(path))
        except (OSError, AttributeError):
            pass
        new_path_parts.append(str(path))

    # PATH에도 prepend — onnxruntime 내부 LoadLibrary가 add_dll_directory를 무시하기 때문
    if new_path_parts:
        existing = os.environ.get("PATH", "")
        os.environ["PATH"] = os.pathsep.join(new_path_parts) + os.pathsep + existing


_add_cuda_dll_directories()
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
MPL_CACHE_DIR = CACHE_DIR / "matplotlib"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))
warnings.filterwarnings("ignore", message=r"`estimate` is deprecated.*", category=FutureWarning)

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from insightface.utils import face_align


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass(frozen=True)
class MemberMeta:
    member_id: str
    group_name: str
    member_name: str


def load_member_meta(csv_path: str | Path) -> dict[str, MemberMeta]:
    path = Path(csv_path)
    if not path.exists():
        return {}

    members: dict[str, MemberMeta] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            member_id = (row.get("member_id") or "").strip()
            if not member_id:
                continue
            members[member_id] = MemberMeta(
                member_id=member_id,
                group_name=(row.get("group_name") or "").strip(),
                member_name=(row.get("member_name") or "").strip(),
            )
    return members


def to_rel_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT_DIR.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def unicode_imread(path: Path) -> np.ndarray | None:
    try:
        buffer = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if buffer.size == 0:
        return None
    return cv2.imdecode(buffer, cv2.IMREAD_COLOR)


def unicode_imwrite(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    extension = ".jpg" if suffix in {"", ".jpeg", ".jpg"} else suffix
    if extension == ".jpg":
        success, encoded = cv2.imencode(extension, image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    else:
        success, encoded = cv2.imencode(extension, image)
    if not success:
        raise OSError(f"Failed to encode image for {path}")
    encoded.tofile(str(path))


def iter_image_paths(input_dir: str | Path, member_ids: set[str] | None = None) -> list[Path]:
    root = Path(input_dir)
    if not root.exists():
        return []

    paths: list[Path] = []
    for member_dir in sorted(item for item in root.iterdir() if item.is_dir()):
        if member_ids and member_dir.name not in member_ids:
            continue
        for image_path in sorted(item for item in member_dir.iterdir() if item.is_file()):
            if image_path.suffix.lower() in IMAGE_SUFFIXES:
                paths.append(image_path)
    return paths


def load_processed_image_paths(csv_path: str | Path) -> set[str]:
    """기존 CSV에 이미 처리된 image_path 집합. 단, 디스크에 파일이 남아있는 것만."""
    path = Path(csv_path)
    if not path.exists():
        return set()

    processed: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            image_path = (row.get("image_path") or "").strip()
            if not image_path:
                continue
            # 디스크에 파일이 없으면 처리됨 캐시에서 제외 → 같은 path 새 파일 들어오면 재처리
            abs_path = (ROOT_DIR / image_path) if not Path(image_path).is_absolute() else Path(image_path)
            if abs_path.exists():
                processed.add(image_path)
    return processed


def _drop_member_rows_from_csv(csv_path: Path, member_ids: set[str]) -> None:
    """CSV에서 지정된 member_id 행만 삭제하고 나머지는 보존."""
    if not csv_path.exists() or not member_ids:
        return
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        kept = [row for row in reader if (row.get("member_id") or "").strip() not in member_ids]
    if not fieldnames:
        return
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept)


def init_face_analyzer(
    model_name: str,
    insightface_root: str | Path,
    ctx_id: int,
    det_thresh: float,
    det_size: int,
    enable_genderage: bool = False,
) -> FaceAnalysis:
    """InsightFace FaceAnalysis 초기화.

    - buffalo_l: RetinaFace + ArcFace r50 + (genderage 선택)
    - antelopev2: SCRFD + ArcFace r100 — 임베딩 품질 더 좋음
    """
    modules = ["detection", "recognition"]
    if enable_genderage:
        modules.append("genderage")
    try:
        kwargs: dict[str, object] = {}
        if ctx_id < 0:
            kwargs["providers"] = ["CPUExecutionProvider"]
        else:
            kwargs["providers"] = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        app = FaceAnalysis(
            name=model_name,
            root=str(insightface_root),
            allowed_modules=modules,
            **kwargs,
        )
        app.prepare(ctx_id=ctx_id, det_thresh=det_thresh, det_size=(det_size, det_size))
        return app
    except Exception as exc:  # pragma: no cover - model availability depends on runtime
        message = (
            "Failed to initialize insightface models. "
            "If this is the first run, the model pack may need to be downloaded.\n"
            f"Suggested retry: python -m src.extract_faces --insightface-root {Path(insightface_root).as_posix()}"
        )
        raise RuntimeError(f"{message}\nOriginal error: {exc}") from exc


def bbox_metrics(face: object, image_shape: tuple[int, int, int]) -> tuple[list[float], float, float, float]:
    bbox = np.asarray(face.bbox, dtype=np.float32)
    x1, y1, x2, y2 = bbox.tolist()
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    image_height, image_width = image_shape[:2]
    area_ratio = (width * height) / max(1.0, float(image_width * image_height))
    return [round(value, 2) for value in [x1, y1, x2, y2]], width, height, area_ratio


def compute_quality_score(det_score: float, face_area_ratio: float) -> float:
    size_score = min(1.0, face_area_ratio * 10.0)
    return round(float(det_score) * size_score, 4)


def build_crop(image: np.ndarray, face: object, crop_size: int) -> np.ndarray:
    if getattr(face, "kps", None) is not None:
        return face_align.norm_crop(image, face.kps, image_size=crop_size)

    bbox = np.asarray(face.bbox, dtype=np.int32)
    x1, y1, x2, y2 = bbox.tolist()
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image.shape[1], x2)
    y2 = min(image.shape[0], y2)
    cropped = image[y1:y2, x1:x2]
    if cropped.size == 0:
        raise ValueError("Face crop is empty after bbox clipping.")
    return cv2.resize(cropped, (crop_size, crop_size), interpolation=cv2.INTER_AREA)


def build_crop_path(image_path: Path, crop_dir: str | Path) -> Path:
    member_id = image_path.parent.name
    return Path(crop_dir) / member_id / f"{image_path.stem}_face.jpg"


def remove_file_if_exists(path: Path) -> bool:
    try:
        path.unlink()
        return True
    except FileNotFoundError:
        return False


def prune_empty_directories(start_dir: Path, stop_dir: Path) -> None:
    current = start_dir
    stop_resolved = stop_dir.resolve()
    while True:
        try:
            current_resolved = current.resolve()
        except FileNotFoundError:
            break

        if current_resolved == stop_resolved:
            break

        try:
            current.rmdir()
        except OSError:
            break
        current = current.parent


def append_rows(csv_path: str | Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return

    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "member_id",
        "group_name",
        "member_name",
        "image_path",
        "crop_path",
        "vector_json",
        "is_valid_face",
        "face_count",
        "quality_score",
        "det_score",
        "face_area_ratio",
        "face_bbox_json",
        "embedding_dim",
        "pose_yaw",
        "pose_pitch",
        "pose_roll",
        "gender",
        "age",
        "error",
    ]
    file_exists = path.exists()

    # 헤더 검사: 기존 파일의 헤더가 현재 fieldnames와 다르면 전체 rewrite (마이그레이션)
    if file_exists:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            existing_fieldnames = list(reader.fieldnames or [])
            if existing_fieldnames != fieldnames:
                existing_rows = list(reader)
                # 기존 row 에 새 컬럼 기본값 채우기 & 사라진 컬럼 제거
                with path.open("w", encoding="utf-8", newline="") as rewrite:
                    writer = csv.DictWriter(rewrite, fieldnames=fieldnames, extrasaction="ignore")
                    writer.writeheader()
                    for existing in existing_rows:
                        out = {k: (existing.get(k) or "") for k in fieldnames}
                        writer.writerow(out)

    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


def process_images(
    image_paths: list[Path],
    member_meta: dict[str, MemberMeta],
    input_dir: str | Path,
    output_csv: str | Path,
    crop_dir: str | Path,
    model_name: str,
    insightface_root: str | Path,
    ctx_id: int,
    det_thresh: float,
    det_size: int,
    crop_size: int,
    min_face_size: int,
    overwrite: bool,
    delete_multi_face: bool,
    delete_no_face: bool,
    clear_crop_dir: bool,
    selected_member_ids: set[str] | None = None,
    min_det_score: float = 0.0,
    enable_genderage: bool = False,
    multi_face_strategy: str = "reject",
) -> tuple[int, int, int]:
    input_dir_path = Path(input_dir)
    output_csv_path = Path(output_csv)
    crop_dir_path = Path(crop_dir)

    if overwrite:
        if selected_member_ids:
            # 선택된 멤버의 행만 제거하고 나머지는 보존
            _drop_member_rows_from_csv(output_csv_path, selected_member_ids)
            if clear_crop_dir:
                for mid in selected_member_ids:
                    member_crop_dir = crop_dir_path / mid
                    if member_crop_dir.exists():
                        shutil.rmtree(member_crop_dir)
        else:
            if output_csv_path.exists():
                output_csv_path.unlink()
            if clear_crop_dir and crop_dir_path.exists():
                shutil.rmtree(crop_dir_path)
        processed_paths = load_processed_image_paths(output_csv_path)
    else:
        processed_paths = load_processed_image_paths(output_csv_path)

    analyzer = init_face_analyzer(
        model_name=model_name,
        insightface_root=insightface_root,
        ctx_id=ctx_id,
        det_thresh=det_thresh,
        det_size=det_size,
        enable_genderage=enable_genderage,
    )

    rows_to_append: list[dict[str, str]] = []
    valid_count = 0
    processed_count = 0
    deleted_multi_face_count = 0
    seen_hashes: dict[str, set[str]] = {}  # member_id → set of hash prefixes

    for image_path in image_paths:
        rel_image_path = to_rel_path(image_path)
        if rel_image_path in processed_paths:
            continue

        # 같은 멤버 폴더 안 hash prefix 중복 검출 → 중복은 즉시 삭제
        member_id = image_path.parent.name
        stem = image_path.stem
        if "_" in stem:
            hash_prefix = stem.split("_", 1)[1]
            member_hashes = seen_hashes.setdefault(member_id, set())
            if hash_prefix in member_hashes:
                if remove_file_if_exists(image_path):
                    prune_empty_directories(image_path.parent, input_dir_path)
                continue
            member_hashes.add(hash_prefix)

        processed_count += 1
        member_id = image_path.parent.name
        meta = member_meta.get(member_id, MemberMeta(member_id=member_id, group_name="", member_name=""))
        crop_path = build_crop_path(image_path, crop_dir)
        remove_file_if_exists(crop_path)
        row = {
            "member_id": meta.member_id,
            "group_name": meta.group_name,
            "member_name": meta.member_name,
            "image_path": rel_image_path,
            "crop_path": "",
            "vector_json": "",
            "is_valid_face": "false",
            "face_count": "0",
            "quality_score": "0.0",
            "det_score": "0.0",
            "face_area_ratio": "0.0",
            "face_bbox_json": "",
            "embedding_dim": "0",
            "pose_yaw": "",
            "pose_pitch": "",
            "pose_roll": "",
            "gender": "",
            "age": "",
            "error": "",
        }

        try:
            image = unicode_imread(image_path)
            if image is None:
                raise ValueError("Unable to decode image.")

            faces = analyzer.get(image)
            row["face_count"] = str(len(faces))
            if not faces:
                raise ValueError("No face detected.")
            if len(faces) > 1:
                if multi_face_strategy == "largest":
                    # 팬캠 모드: 가장 큰 얼굴만 선택 (포커스 대상은 항상 가장 크게 찍힘)
                    def _face_area(f: object) -> float:
                        b = np.asarray(f.bbox, dtype=np.float32)
                        return max(0.0, (b[2] - b[0]) * (b[3] - b[1]))
                    face = max(faces, key=_face_area)
                elif multi_face_strategy == "center":
                    image_center = np.array([image.shape[1] / 2.0, image.shape[0] / 2.0])
                    def _dist_from_center(f: object) -> float:
                        b = np.asarray(f.bbox, dtype=np.float32)
                        face_center = np.array([(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0])
                        return float(np.linalg.norm(face_center - image_center))
                    face = min(faces, key=_dist_from_center)
                else:
                    raise ValueError(f"Multiple faces detected: {len(faces)}")
            else:
                face = faces[0]

            bbox_json, face_width, face_height, face_area_ratio = bbox_metrics(face, image.shape)
            face_det_score = float(face.det_score or 0.0)
            row["det_score"] = f"{face_det_score:.4f}"
            row["face_area_ratio"] = f"{face_area_ratio:.4f}"
            row["face_bbox_json"] = json.dumps(bbox_json, ensure_ascii=True)

            if face_det_score < float(min_det_score):
                raise ValueError(
                    f"det_score {face_det_score:.3f} below min {min_det_score}"
                )

            if min(face_width, face_height) < float(min_face_size):
                raise ValueError(
                    f"Detected face too small: {int(face_width)}x{int(face_height)} < {min_face_size}px"
                )

            if getattr(face, "embedding", None) is None:
                raise ValueError("Embedding was not generated for the detected face.")

            crop = build_crop(image, face, crop_size=crop_size)
            unicode_imwrite(crop_path, crop)

            embedding = np.asarray(face.embedding, dtype=np.float32)
            row["crop_path"] = to_rel_path(crop_path)
            row["vector_json"] = json.dumps(embedding.tolist(), ensure_ascii=True, separators=(",", ":"))
            row["is_valid_face"] = "true"
            row["quality_score"] = f"{compute_quality_score(float(face.det_score or 0.0), face_area_ratio):.4f}"
            row["embedding_dim"] = str(int(embedding.size))

            pose = getattr(face, "pose", None)
            if pose is not None:
                try:
                    row["pose_yaw"] = f"{float(pose[0]):.2f}"
                    row["pose_pitch"] = f"{float(pose[1]):.2f}"
                    row["pose_roll"] = f"{float(pose[2]):.2f}"
                except (TypeError, IndexError, ValueError):
                    pass
            sex = getattr(face, "sex", None)
            if sex is not None:
                row["gender"] = str(sex)
            age = getattr(face, "age", None)
            if age is not None:
                try:
                    row["age"] = f"{float(age):.1f}"
                except (TypeError, ValueError):
                    pass
            valid_count += 1
        except Exception as exc:
            row["error"] = str(exc)
            face_count = int(row["face_count"]) if row["face_count"].isdigit() else 0
            err_msg = str(exc)
            is_low_det = "det_score" in err_msg and "below min" in err_msg
            should_delete = (
                (delete_multi_face and face_count > 1)
                or (delete_no_face and face_count == 0)
                or (is_low_det and min_det_score > 0)  # 저품질도 자동 삭제
            )
            if should_delete:
                if remove_file_if_exists(image_path):
                    deleted_multi_face_count += 1
                    prune_empty_directories(image_path.parent, input_dir_path)
                continue

        rows_to_append.append(row)
        if len(rows_to_append) >= 50:
            append_rows(output_csv_path, rows_to_append)
            rows_to_append.clear()

    append_rows(output_csv_path, rows_to_append)
    return processed_count, valid_count, deleted_multi_face_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract face crops and embeddings from crawled idol images.")
    parser.add_argument(
        "--members",
        default="data/members.csv",
        help="Path to the members CSV for metadata lookup.",
    )
    parser.add_argument(
        "--input-dir",
        default="data/raw_images",
        help="Directory that contains crawled raw images grouped by member_id.",
    )
    parser.add_argument(
        "--output-csv",
        default="data/image_embeddings.csv",
        help="CSV file where image-level embeddings will be stored.",
    )
    parser.add_argument(
        "--crop-dir",
        default="data/face_crops",
        help="Directory where aligned face crops will be written.",
    )
    parser.add_argument(
        "--insightface-root",
        default=str((CACHE_DIR / "insightface").as_posix()),
        help="Directory where insightface model files will be stored.",
    )
    parser.add_argument(
        "--model-name",
        default="buffalo_l",
        help="InsightFace model pack name. 'buffalo_l' (빠름, r50) 또는 'antelopev2' (정확, r100) 권장.",
    )
    parser.add_argument(
        "--enable-genderage",
        action="store_true",
        help="genderage 모듈 활성화해서 성별/나이 추정까지 저장 (모델 로딩 약간 느려짐).",
    )
    parser.add_argument(
        "--multi-face-strategy",
        choices=["reject", "largest", "center"],
        default="reject",
        help=(
            "multi-face 이미지 처리: reject(기본)=폐기, "
            "largest=가장 큰 얼굴 선택 (팬캠 추천), "
            "center=화면 중앙에 가장 가까운 얼굴."
        ),
    )
    parser.add_argument(
        "--ctx-id",
        type=int,
        default=-1,
        help="Execution context for insightface. Use -1 for CPU.",
    )
    parser.add_argument(
        "--det-thresh",
        type=float,
        default=0.5,
        help="Face detection confidence threshold.",
    )
    parser.add_argument(
        "--det-size",
        type=int,
        default=640,
        help="Face detection input size.",
    )
    parser.add_argument(
        "--crop-size",
        type=int,
        default=112,
        help="Aligned crop output size.",
    )
    parser.add_argument(
        "--min-face-size",
        type=int,
        default=80,
        help="Minimum face width and height in pixels to accept.",
    )
    parser.add_argument(
        "--min-det-score",
        type=float,
        default=0.75,
        help="Minimum face detection score to accept (0.75 filters mask/occlusion/profile).",
    )
    parser.add_argument(
        "--member-ids",
        nargs="*",
        help="Optional subset of member IDs to process.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Optional cap on the number of images to process.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild the output CSV from scratch instead of appending new rows only.",
    )
    parser.add_argument(
        "--delete-multi-face",
        action="store_true",
        help="Delete raw images when more than one face is detected in a single image.",
    )
    parser.add_argument(
        "--delete-no-face",
        action="store_true",
        help="Delete raw images when no face is detected.",
    )
    parser.add_argument(
        "--clear-crop-dir",
        action="store_true",
        help="Delete the crop directory before processing when --overwrite is used.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_ids = {member_id.strip() for member_id in (args.member_ids or []) if member_id.strip()} or None
    image_paths = iter_image_paths(args.input_dir, member_ids=selected_ids)
    if args.max_images > 0:
        image_paths = image_paths[: args.max_images]

    if not image_paths:
        raise SystemExit("No raw images found to process.")

    members = load_member_meta(args.members)
    processed_count, valid_count, deleted_multi_face_count = process_images(
        image_paths=image_paths,
        member_meta=members,
        input_dir=args.input_dir,
        output_csv=args.output_csv,
        crop_dir=args.crop_dir,
        model_name=args.model_name,
        insightface_root=args.insightface_root,
        ctx_id=args.ctx_id,
        det_thresh=args.det_thresh,
        det_size=args.det_size,
        crop_size=args.crop_size,
        min_face_size=args.min_face_size,
        overwrite=args.overwrite,
        delete_multi_face=args.delete_multi_face,
        delete_no_face=args.delete_no_face,
        clear_crop_dir=args.clear_crop_dir,
        selected_member_ids=selected_ids,
        min_det_score=args.min_det_score,
        enable_genderage=args.enable_genderage,
        multi_face_strategy=args.multi_face_strategy,
    )
    print(
        f"Processed {processed_count} images. "
        f"Valid faces with embeddings: {valid_count}. "
        f"Deleted multi-face raw images: {deleted_multi_face_count}. "
        f"Output CSV: {args.output_csv}"
    )


if __name__ == "__main__":
    main()
