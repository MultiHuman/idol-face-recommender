"""YouTube 팬캠에서 멤버별 얼굴 키프레임 샘플링.

왜: DDGS 검색만으론 조명·각도·연도가 단조롭다. 직캠은 한 영상에서
수십 가지 각도·표정을 얻을 수 있어 데이터 다양성이 근본적으로 해결됨.

흐름:
1) yt-dlp search: `ytsearch{N}:{query}` 로 팬캠 URL 후보 찾음
2) yt-dlp 로 360p 다운로드 (저해상도로 충분 — 얼굴은 crop)
3) ffmpeg 로 N 초마다 1프레임 추출 → jpg
4) pHash 중복 제거, 해상도/blur 필터 (crawl.py 의 기준 재사용)
5) data/raw_images/{member_id}/ 에 기존 명명 규칙(`NNN_<sha_prefix>.jpg`)으로 저장
6) manifest.csv 에 source_url=YouTube ID 로 로그

필요: yt-dlp, ffmpeg (PATH). ffmpeg 없으면 yt_dlp 가 자동 다운로드 시도.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import shutil
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path

import cv2
import imagehash
import numpy as np
from PIL import Image

from src.crawl import (
    DEFAULT_QUERY_TEMPLATES,
    MANIFEST_FIELDNAMES,
    _append_manifest_rows,
    _blur_variance,
    _existing_file_count,
    _load_global_phashes,
    _load_manifest_urls,
    _load_member_digests,
    _phash_is_dup,
    build_search_queries,
    load_members,
)


def _which_ffmpeg() -> str | None:
    return shutil.which("ffmpeg")


def _ytdlp_search(query: str, max_videos: int) -> list[dict[str, str]]:
    import yt_dlp

    opts = {
        "quiet": True,
        "skip_download": True,
        "extract_flat": "in_playlist",
        "default_search": "ytsearch",
        "noplaylist": False,
    }
    search_url = f"ytsearch{max_videos}:{query}"
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(search_url, download=False)
    entries = (info or {}).get("entries") or []
    return [
        {
            "id": str(e.get("id") or ""),
            "url": str(e.get("url") or e.get("webpage_url") or ""),
            "title": str(e.get("title") or ""),
            "duration": str(e.get("duration") or ""),
        }
        for e in entries
        if e
    ]


def _ytdlp_download(url: str, out_path: Path, resolution_cap: int) -> bool:
    import yt_dlp

    opts = {
        "quiet": True,
        "outtmpl": str(out_path.with_suffix(".%(ext)s")),
        "format": f"bestvideo[height<={resolution_cap}]+bestaudio/best[height<={resolution_cap}]/best",
        "merge_output_format": "mp4",
        "noplaylist": True,
        "concurrent_fragment_downloads": 4,
        "retries": 2,
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])
    except Exception as exc:
        print(f"  [warn] yt-dlp failed for {url}: {exc}")
        return False
    # 확장자 찾기
    for cand in out_path.parent.glob(out_path.stem + ".*"):
        if cand.suffix.lower() in {".mp4", ".mkv", ".webm"}:
            if cand != out_path.with_suffix(cand.suffix):
                cand.rename(out_path.with_suffix(cand.suffix))
            return True
    return False


def _extract_keyframes_ffmpeg(
    video_path: Path,
    out_dir: Path,
    every_seconds: float,
    max_frames: int,
) -> list[Path]:
    ffmpeg = _which_ffmpeg()
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(out_dir / "frame_%05d.jpg")
    # every_seconds 당 1프레임, 최대 max_frames
    cmd = [
        ffmpeg or "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vf",
        f"fps=1/{every_seconds}",
        "-frames:v",
        str(max_frames),
        "-q:v",
        "3",
        pattern,
    ]
    try:
        subprocess.run(cmd, check=True, timeout=600)
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        print(f"  [warn] ffmpeg failed: {exc}")
        return []
    return sorted(out_dir.glob("frame_*.jpg"))


def _frame_passes_quality(
    frame_path: Path,
    min_side: int,
    min_blur: float,
) -> tuple[Image.Image, str, int, int, float] | None:
    try:
        pil = Image.open(frame_path).convert("RGB")
    except Exception:
        return None
    width, height = pil.size
    if min(width, height) < min_side:
        return None
    try:
        ph = str(imagehash.phash(pil))
    except Exception:
        ph = ""
    blur = _blur_variance(pil)
    if min_blur > 0 and blur < min_blur:
        return None
    return pil, ph, width, height, blur


def _save_frame(
    pil: Image.Image,
    member_dir: Path,
    next_index: int,
) -> tuple[Path, str, int]:
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=92)
    data = buf.getvalue()
    digest = hashlib.sha256(data).hexdigest()
    digest_prefix = digest[:12]
    file_name = f"{next_index:03d}_{digest_prefix}.jpg"
    path = member_dir / file_name
    path.write_bytes(data)
    return path, digest, len(data)


def run(
    members_csv: Path,
    output_dir: Path,
    manifest_path: Path,
    limit_per_member: int,
    videos_per_member: int,
    frames_per_video: int,
    frame_every_seconds: float,
    resolution_cap: int,
    min_side: int,
    min_blur: float,
    phash_threshold: int,
    delay_seconds: float,
    member_ids: list[str] | None,
    query_templates: tuple[str, ...],
) -> None:
    if _which_ffmpeg() is None:
        print("[warn] ffmpeg not found in PATH. yt-dlp bundled ffmpeg may work, but keyframe extraction will likely fail.")

    members = load_members(members_csv)
    if member_ids:
        allowed = {m.strip() for m in member_ids if m.strip()}
        members = [m for m in members if m.member_id in allowed]

    output_dir.mkdir(parents=True, exist_ok=True)
    seen_urls = _load_manifest_urls(manifest_path)
    global_phash_cache = _load_global_phashes(manifest_path)

    for member in members:
        member_dir = output_dir / member.member_id
        member_dir.mkdir(parents=True, exist_ok=True)
        existing_count = _existing_file_count(member_dir)
        if existing_count >= limit_per_member:
            print(f"[skip] {member.member_id}: already has {existing_count} files")
            continue

        # 직캠 전용 쿼리 오버라이드가 없으면 기본 템플릿 중 '직캠', 'fancam' 계열만 씀
        queries = [q for q in build_search_queries(member, query_templates)
                   if any(kw in q.lower() for kw in ("fancam", "직캠", "perfcam", "focus"))]
        if not queries:
            # 그래도 안 생기면 기본 hint + 'fancam' 강제 조합
            queries = [f"{member.search_hint} fancam"]
        print(f"[fancam] {member.member_id}: queries={queries}")

        local_phashes: list[imagehash.ImageHash] = []
        for path in member_dir.iterdir():
            if not path.is_file():
                continue
            try:
                pil = Image.open(path).convert("RGB")
                local_phashes.append(imagehash.hex_to_hash(str(imagehash.phash(pil))))
            except Exception:
                continue

        videos_collected = 0
        rows_to_append: list[dict[str, str]] = []
        next_index = existing_count + 1

        for query in queries:
            if videos_collected >= videos_per_member or existing_count >= limit_per_member:
                break
            try:
                candidates = _ytdlp_search(query, videos_per_member * 2)
            except Exception as exc:
                print(f"  [warn] search failed '{query}': {exc}")
                continue

            for vid in candidates:
                if videos_collected >= videos_per_member or existing_count >= limit_per_member:
                    break
                video_url = vid["url"] or f"https://www.youtube.com/watch?v={vid['id']}"
                if video_url in seen_urls:
                    continue

                print(f"  [video] {vid['title'][:60]!r}  {video_url}")
                with tempfile.TemporaryDirectory(prefix="fancam_") as tmp:
                    tmp_path = Path(tmp)
                    video_stem = tmp_path / vid["id"]
                    if not _ytdlp_download(video_url, video_stem, resolution_cap):
                        seen_urls.add(video_url)
                        continue

                    video_file = next(
                        (p for p in tmp_path.glob(vid["id"] + ".*")
                         if p.suffix.lower() in {".mp4", ".mkv", ".webm"}),
                        None,
                    )
                    if video_file is None:
                        seen_urls.add(video_url)
                        continue

                    frames_dir = tmp_path / "frames"
                    frames = _extract_keyframes_ffmpeg(
                        video_file, frames_dir, frame_every_seconds, frames_per_video
                    )
                    if not frames:
                        seen_urls.add(video_url)
                        continue

                    seen_urls.add(video_url)
                    videos_collected += 1

                    kept = 0
                    for fp in frames:
                        if existing_count >= limit_per_member:
                            break
                        inspected = _frame_passes_quality(fp, min_side, min_blur)
                        if inspected is None:
                            continue
                        pil, ph, w, h, blur = inspected
                        if _phash_is_dup(ph, local_phashes, phash_threshold):
                            continue
                        if ph and ph in global_phash_cache:
                            continue
                        saved_path, digest, size = _save_frame(pil, member_dir, next_index)
                        next_index += 1
                        existing_count += 1
                        kept += 1
                        if ph:
                            try:
                                h_obj = imagehash.hex_to_hash(ph)
                                local_phashes.append(h_obj)
                                global_phash_cache[ph] = h_obj
                            except (ValueError, TypeError):
                                pass
                        rows_to_append.append({
                            "downloaded_at": datetime.now().isoformat(timespec="seconds"),
                            "member_id": member.member_id,
                            "group_name": member.group_name,
                            "member_name": member.member_name,
                            "query": query,
                            "source_url": f"{video_url}#frame={fp.name}",
                            "thumbnail_url": "",
                            "source_page": video_url,
                            "source_title": vid["title"],
                            "source_description": "",
                            "file_path": str(saved_path),
                            "sha256": digest,
                            "phash": ph,
                            "bytes": str(size),
                            "width": str(w),
                            "height": str(h),
                            "blur_score": f"{blur:.2f}",
                            "status": "downloaded",
                            "error": "",
                        })
                    print(f"    kept {kept}/{len(frames)} frames (total={existing_count})")

            if delay_seconds > 0:
                time.sleep(delay_seconds)

        _append_manifest_rows(manifest_path, rows_to_append)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect face keyframes from YouTube fancams.")
    parser.add_argument("--members", default="data/members.csv")
    parser.add_argument("--output-dir", default="data/raw_images")
    parser.add_argument("--manifest", default="data/raw_images/manifest.csv")
    parser.add_argument("--limit-per-member", type=int, default=30)
    parser.add_argument("--videos-per-member", type=int, default=3)
    parser.add_argument("--frames-per-video", type=int, default=12)
    parser.add_argument("--frame-every-seconds", type=float, default=8.0)
    parser.add_argument("--resolution-cap", type=int, default=480)
    parser.add_argument("--min-side", type=int, default=400)
    parser.add_argument("--min-blur", type=float, default=80.0)
    parser.add_argument("--phash-threshold", type=int, default=6)
    parser.add_argument("--delay-seconds", type=float, default=2.0)
    parser.add_argument("--member-ids", nargs="*")
    parser.add_argument(
        "--query-template",
        action="append",
        default=None,
        help="쿼리 템플릿을 덮어쓴다. {hint}/{group_en}/{member_en}. 기본은 crawl.py 와 동일한 7종.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    templates = tuple(args.query_template) if args.query_template else DEFAULT_QUERY_TEMPLATES
    run(
        members_csv=Path(args.members),
        output_dir=Path(args.output_dir),
        manifest_path=Path(args.manifest),
        limit_per_member=args.limit_per_member,
        videos_per_member=args.videos_per_member,
        frames_per_video=args.frames_per_video,
        frame_every_seconds=args.frame_every_seconds,
        resolution_cap=args.resolution_cap,
        min_side=args.min_side,
        min_blur=args.min_blur,
        phash_threshold=args.phash_threshold,
        delay_seconds=args.delay_seconds,
        member_ids=args.member_ids,
        query_templates=templates,
    )


if __name__ == "__main__":
    main()
