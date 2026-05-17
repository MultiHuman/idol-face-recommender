from __future__ import annotations

import argparse
import csv
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import mimetypes
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse, urlsplit, urlunsplit
from urllib.request import Request, urlopen

import cv2
import imagehash
import numpy as np
from PIL import Image

from ddgs import DDGS


USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/135.0.0.0 Safari/537.36"
)

IMAGE_EXTENSIONS = {
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "image/bmp": ".bmp",
}

DOWNLOAD_BATCH_MULTIPLIER = 2
PHASH_HAMMING_THRESHOLD = 6  # dHash/pHash 해밍거리 ≤ 6 이면 near-duplicate로 취급
DEFAULT_MAX_BYTES = 15_000_000
DEFAULT_QUERY_TEMPLATES: tuple[str, ...] = (
    "{hint}",
    "{hint} 고화질",
    "{hint} 직캠",
    "{hint} 셀카",
    "{hint} 포토",
    "{group_en} {member_en} photoshoot",
    "{group_en} {member_en} fancam",
)

MANIFEST_FIELDNAMES = [
    "downloaded_at",
    "member_id",
    "group_name",
    "member_name",
    "query",
    "source_url",
    "thumbnail_url",
    "source_page",
    "source_title",
    "source_description",
    "file_path",
    "sha256",
    "phash",
    "bytes",
    "width",
    "height",
    "blur_score",
    "status",
    "error",
]


@dataclass(frozen=True)
class MemberRecord:
    member_id: str
    group_name: str
    member_name: str
    search_hint: str
    include_terms: tuple[str, ...]
    exclude_terms: tuple[str, ...]


@dataclass(frozen=True)
class CandidateImage:
    image_url: str
    thumbnail_url: str
    source_page: str
    title: str
    description: str
    query: str


@dataclass(frozen=True)
class DownloadResult:
    downloaded_from: str
    image_bytes: bytes
    content_type: str | None
    width: int
    height: int
    phash: str
    blur_score: float


def _split_terms(value: str) -> tuple[str, ...]:
    return tuple(term.strip() for term in value.split("|") if term.strip())


def load_members(csv_path: str | Path) -> list[MemberRecord]:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Members CSV does not exist: {path}")

    rows: list[MemberRecord] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            member_id = (row.get("member_id") or "").strip()
            if not member_id:
                continue

            group_name = (row.get("group_name") or "").strip()
            member_name = (row.get("member_name") or "").strip()
            search_hint = (row.get("search_hint") or "").strip()
            if not search_hint:
                search_hint = " ".join(part for part in [group_name, member_name] if part).strip()
            include_terms = _split_terms((row.get("include_terms") or "").strip())
            exclude_terms = _split_terms((row.get("exclude_terms") or "").strip())

            rows.append(
                MemberRecord(
                    member_id=member_id,
                    group_name=group_name,
                    member_name=member_name,
                    search_hint=search_hint,
                    include_terms=include_terms,
                    exclude_terms=exclude_terms,
                )
            )

    return rows


def _quote_query_term(term: str) -> str:
    return f"\"{term}\"" if " " in term else term


def build_search_queries(member: MemberRecord, templates: Iterable[str]) -> list[str]:
    """멤버별로 여러 쿼리를 생성. 같은 얼굴의 조명·각도·연도 분산을 노린다."""
    queries: list[str] = []
    seen: set[str] = set()
    for template in templates:
        try:
            base = template.format(
                hint=member.search_hint.strip(),
                group_en=member.group_name.strip(),
                member_en=member.member_name.strip(),
            ).strip()
        except KeyError:
            continue
        if not base:
            continue

        parts = [base]
        lowered = base.lower()
        for term in member.include_terms:
            if term.lower() not in lowered:
                parts.append(_quote_query_term(term))
        for term in member.exclude_terms:
            parts.append(f"-{_quote_query_term(term)}")

        query = " ".join(p for p in parts if p).strip()
        if query and query not in seen:
            seen.add(query)
            queries.append(query)
    return queries


def _extract_candidates(query: str, max_candidates: int) -> list[CandidateImage]:
    with DDGS() as ddgs:
        results = ddgs.images(query, max_results=max_candidates)

    candidates: list[CandidateImage] = []
    for item in results or []:
        image_url = (item.get("image") or "").strip()
        if not image_url or not image_url.startswith("http"):
            continue
        candidates.append(
            CandidateImage(
                image_url=image_url,
                thumbnail_url=(item.get("thumbnail") or "").strip(),
                source_page=(item.get("url") or "").strip(),
                title=(item.get("title") or "").strip(),
                description="",
                query=query,
            )
        )
    return candidates


def collect_candidates(
    member: MemberRecord,
    queries: list[str],
    max_per_query: int,
) -> list[CandidateImage]:
    seen: set[str] = set()
    out: list[CandidateImage] = []
    for q in queries:
        try:
            batch = _extract_candidates(q, max_per_query)
        except Exception as exc:
            print(f"  [warn] query failed '{q}': {exc}")
            continue
        for c in batch:
            if c.image_url in seen:
                continue
            seen.add(c.image_url)
            out.append(c)
    return out


def _normalize_for_match(value: str) -> str:
    return " ".join(value.lower().split())


def _candidate_matches(member: MemberRecord, candidate: CandidateImage) -> bool:
    searchable_text = _normalize_for_match(
        " ".join(
            [
                candidate.image_url,
                candidate.thumbnail_url,
                candidate.source_page,
                candidate.title,
                candidate.description,
            ]
        )
    )

    if member.exclude_terms:
        for term in member.exclude_terms:
            if _normalize_for_match(term) in searchable_text:
                return False

    if member.include_terms:
        return any(_normalize_for_match(term) in searchable_text for term in member.include_terms)

    return True


def _guess_extension(url: str, content_type: str | None) -> str:
    content_type = (content_type or "").split(";")[0].strip().lower()
    if content_type in IMAGE_EXTENSIONS:
        return IMAGE_EXTENSIONS[content_type]

    parsed = urlparse(url)
    suffix = Path(parsed.path).suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}:
        return ".jpg" if suffix == ".jpeg" else suffix

    guessed, _ = mimetypes.guess_type(url)
    if guessed in IMAGE_EXTENSIONS:
        return IMAGE_EXTENSIONS[guessed]

    return ".jpg"


def _normalize_url_for_request(url: str) -> str:
    parts = urlsplit(url)
    netloc = parts.netloc.encode("idna").decode("ascii")
    path = quote(parts.path, safe="/%:@")
    query = quote(parts.query, safe="=&?/%:@,+")
    fragment = quote(parts.fragment, safe="")
    return urlunsplit((parts.scheme, netloc, path, query, fragment))


def _referer_for(url: str, source_page: str) -> str:
    """CDN이 자사 도메인 referer를 요구하는 경우가 많으니 source_page가 있으면 그 origin을 쓴다."""
    target = source_page.strip() or url
    parts = urlsplit(target)
    if not parts.scheme or not parts.netloc:
        return "https://duckduckgo.com/"
    return f"{parts.scheme}://{parts.netloc}/"


def _download_bytes(
    url: str,
    referer: str,
    timeout_seconds: int,
    max_bytes: int,
) -> tuple[bytes, str | None]:
    normalized_url = _normalize_url_for_request(url)
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
        "Referer": referer,
    }
    request = Request(normalized_url, headers=headers)
    with urlopen(request, timeout=timeout_seconds) as response:
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = response.read(64 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if max_bytes > 0 and total > max_bytes:
                raise ValueError(f"image larger than max_bytes {max_bytes}")
            chunks.append(chunk)
        return b"".join(chunks), response.headers.get_content_type()


def _decode_image(image_bytes: bytes) -> Image.Image | None:
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.load()
    except Exception:
        return None
    return image.convert("RGB")


def _blur_variance(pil: Image.Image) -> float:
    """Laplacian variance — 낮을수록 흐릿하다. 임계값 100 미만이면 blur."""
    arr = np.asarray(pil.convert("L"), dtype=np.uint8)
    if arr.size == 0:
        return 0.0
    lap = cv2.Laplacian(arr, cv2.CV_64F)
    return float(lap.var())


def _inspect_image(image_bytes: bytes) -> tuple[Image.Image, int, int, str, float] | None:
    pil = _decode_image(image_bytes)
    if pil is None:
        return None
    width, height = pil.size
    try:
        phash = str(imagehash.phash(pil))
    except Exception:
        phash = ""
    blur = _blur_variance(pil)
    return pil, width, height, phash, blur


def _download_candidate(
    candidate: CandidateImage,
    timeout_seconds: int,
    min_bytes: int,
    max_bytes: int,
    min_side: int,
    min_blur: float,
) -> DownloadResult:
    attempted_urls = [candidate.image_url]
    if candidate.thumbnail_url and candidate.thumbnail_url not in attempted_urls:
        attempted_urls.append(candidate.thumbnail_url)

    referer = _referer_for(candidate.image_url, candidate.source_page)
    last_error: Exception | None = None

    for attempt_url in attempted_urls:
        try:
            image_bytes, content_type = _download_bytes(
                attempt_url,
                referer,
                timeout_seconds=timeout_seconds,
                max_bytes=max_bytes,
            )
            if len(image_bytes) < min_bytes:
                raise ValueError(f"too small: {len(image_bytes)} bytes")

            inspected = _inspect_image(image_bytes)
            if inspected is None:
                raise ValueError("failed to decode image")
            _pil, width, height, phash, blur = inspected

            if min(width, height) < min_side:
                raise ValueError(f"resolution {width}x{height} below min_side {min_side}")
            if min_blur > 0 and blur < min_blur:
                raise ValueError(f"blur variance {blur:.1f} below min {min_blur}")

            return DownloadResult(
                downloaded_from=attempt_url,
                image_bytes=image_bytes,
                content_type=content_type,
                width=width,
                height=height,
                phash=phash,
                blur_score=blur,
            )
        except (HTTPError, URLError, TimeoutError, ValueError, OSError) as exc:
            last_error = exc

    raise ValueError(str(last_error or "download failed"))


def _existing_file_count(directory: Path) -> int:
    return sum(1 for item in directory.iterdir() if item.is_file())


def _load_manifest_urls(manifest_path: Path) -> set[str]:
    if not manifest_path.exists():
        return set()

    seen: set[str] = set()
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            source_url = (row.get("source_url") or "").strip()
            if source_url:
                seen.add(source_url)
    return seen


def _load_member_digests(member_dir: Path) -> set[str]:
    """이미 다운로드된 파일들의 sha256 prefix 집합."""
    digests: set[str] = set()
    if not member_dir.exists():
        return digests
    for path in member_dir.iterdir():
        if not path.is_file():
            continue
        stem = path.stem
        if "_" in stem:
            prefix = stem.split("_", 1)[1]
            if prefix:
                digests.add(prefix)
    return digests


def _load_global_phashes(manifest_path: Path) -> dict[str, imagehash.ImageHash]:
    """manifest의 모든 phash (모든 멤버) — 같은 이미지가 두 멤버에게 붙지 않도록."""
    result: dict[str, imagehash.ImageHash] = {}
    if not manifest_path.exists():
        return result
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            ph = (row.get("phash") or "").strip()
            status = (row.get("status") or "").strip()
            if not ph or status not in {"downloaded", "duplicate_phash"}:
                continue
            try:
                result[ph] = imagehash.hex_to_hash(ph)
            except (ValueError, TypeError):
                continue
    return result


def _phash_is_dup(
    ph: str,
    local_hashes: list[imagehash.ImageHash],
    threshold: int,
) -> bool:
    if not ph or not local_hashes:
        return False
    try:
        probe = imagehash.hex_to_hash(ph)
    except (ValueError, TypeError):
        return False
    for h in local_hashes:
        if probe - h <= threshold:
            return True
    return False


def _append_manifest_rows(manifest_path: Path, rows: Iterable[dict[str, str]]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        return

    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            existing_fieldnames = reader.fieldnames or []
            if existing_fieldnames != MANIFEST_FIELDNAMES:
                existing_rows = list(reader)
                with manifest_path.open("w", encoding="utf-8", newline="") as rewrite_handle:
                    writer = csv.DictWriter(rewrite_handle, fieldnames=MANIFEST_FIELDNAMES)
                    writer.writeheader()
                    for existing_row in existing_rows:
                        writer.writerow({field: existing_row.get(field, "") for field in MANIFEST_FIELDNAMES})
        file_exists = True
    else:
        file_exists = False

    with manifest_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


def crawl_members(
    members: list[MemberRecord],
    output_dir: str | Path,
    manifest_path: str | Path,
    limit_per_member: int,
    max_candidates: int,
    timeout_seconds: int,
    min_bytes: int,
    max_bytes: int,
    min_side: int,
    min_blur: float,
    delay_ms: int,
    download_workers: int,
    query_templates: tuple[str, ...],
    phash_threshold: int,
    dedup_across_members: bool,
) -> None:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = Path(manifest_path)
    seen_urls = _load_manifest_urls(manifest)
    global_phash_cache = _load_global_phashes(manifest) if dedup_across_members else {}

    for member in members:
        member_dir = output_root / member.member_id
        member_dir.mkdir(parents=True, exist_ok=True)
        seen_digests = _load_member_digests(member_dir)
        existing_count = _existing_file_count(member_dir)
        if existing_count >= limit_per_member:
            print(
                f"[skip] {member.member_id}: already has {existing_count} files "
                f"(limit={limit_per_member})"
            )
            continue

        queries = build_search_queries(member, query_templates)
        if not queries:
            print(f"[skip] {member.member_id}: no query produced")
            continue
        print(
            f"[crawl] {member.member_id}: {len(queries)} queries — "
            f"{queries[0]!r}{' + ...' if len(queries) > 1 else ''}"
        )

        # 각 쿼리당 max_candidates 부분을 받아서 합산 (중복 url 자동 제외)
        per_query = max(8, max_candidates // max(1, len(queries)))
        candidates = collect_candidates(member, queries, per_query)
        if not candidates:
            print(f"[warn] {member.member_id}: no candidates")
            continue

        local_member_phashes: list[imagehash.ImageHash] = []
        # 기존 이미지의 phash도 로드 (재실행 안전성)
        for path in member_dir.iterdir():
            if not path.is_file():
                continue
            try:
                pil = Image.open(path).convert("RGB")
                ph = str(imagehash.phash(pil))
                local_member_phashes.append(imagehash.hex_to_hash(ph))
            except Exception:
                continue

        rows_to_append: list[dict[str, str]] = []
        download_count = 0
        filtered_count = 0
        download_candidates: list[CandidateImage] = []
        for candidate in candidates:
            if not _candidate_matches(member, candidate):
                filtered_count += 1
                continue
            if candidate.image_url in seen_urls:
                continue
            download_candidates.append(candidate)

        next_file_index = existing_count + 1
        candidate_index = 0
        while candidate_index < len(download_candidates) and existing_count < limit_per_member:
            remaining_slots = limit_per_member - existing_count
            batch_size = min(
                len(download_candidates) - candidate_index,
                max(1, remaining_slots * DOWNLOAD_BATCH_MULTIPLIER),
            )
            batch = download_candidates[candidate_index : candidate_index + batch_size]
            candidate_index += batch_size
            worker_count = min(download_workers, len(batch))
            if worker_count <= 0:
                break

            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                future_to_candidate = {
                    executor.submit(
                        _download_candidate,
                        candidate,
                        timeout_seconds,
                        min_bytes,
                        max_bytes,
                        min_side,
                        min_blur,
                    ): candidate
                    for candidate in batch
                }

                for future in as_completed(future_to_candidate):
                    candidate = future_to_candidate[future]
                    downloaded_at = datetime.now().isoformat(timespec="seconds")
                    row = {
                        "downloaded_at": downloaded_at,
                        "member_id": member.member_id,
                        "group_name": member.group_name,
                        "member_name": member.member_name,
                        "query": candidate.query,
                        "source_url": candidate.image_url,
                        "thumbnail_url": candidate.thumbnail_url,
                        "source_page": candidate.source_page,
                        "source_title": candidate.title,
                        "source_description": candidate.description,
                        "file_path": "",
                        "sha256": "",
                        "phash": "",
                        "bytes": "0",
                        "width": "0",
                        "height": "0",
                        "blur_score": "0.0",
                        "status": "failed",
                        "error": "",
                    }

                    try:
                        result = future.result()
                        if existing_count >= limit_per_member:
                            continue

                        row["phash"] = result.phash
                        row["width"] = str(result.width)
                        row["height"] = str(result.height)
                        row["blur_score"] = f"{result.blur_score:.2f}"

                        digest = hashlib.sha256(result.image_bytes).hexdigest()
                        digest_prefix = digest[:12]
                        row["sha256"] = digest
                        row["bytes"] = str(len(result.image_bytes))

                        if digest_prefix in seen_digests:
                            row["status"] = "duplicate_sha"
                            row["error"] = "duplicate content hash"
                            seen_urls.add(candidate.image_url)
                            if candidate.thumbnail_url:
                                seen_urls.add(candidate.thumbnail_url)
                            rows_to_append.append(row)
                            continue

                        # pHash 멤버 내 중복
                        if _phash_is_dup(result.phash, local_member_phashes, phash_threshold):
                            row["status"] = "duplicate_phash"
                            row["error"] = f"phash within {phash_threshold} of existing"
                            seen_urls.add(candidate.image_url)
                            rows_to_append.append(row)
                            continue

                        # pHash cross-member 중복 (옵션)
                        if dedup_across_members and result.phash and result.phash in global_phash_cache:
                            row["status"] = "duplicate_crossmember"
                            row["error"] = "same phash seen in another member"
                            seen_urls.add(candidate.image_url)
                            rows_to_append.append(row)
                            continue

                        extension = _guess_extension(result.downloaded_from, result.content_type)
                        file_name = f"{next_file_index:03d}_{digest_prefix}{extension}"
                        file_path = member_dir / file_name
                        file_path.write_bytes(result.image_bytes)

                        row["source_url"] = result.downloaded_from
                        row["file_path"] = str(file_path)
                        row["status"] = "downloaded"
                        seen_urls.add(candidate.image_url)
                        seen_digests.add(digest_prefix)
                        if result.phash:
                            try:
                                local_member_phashes.append(imagehash.hex_to_hash(result.phash))
                                if dedup_across_members:
                                    global_phash_cache[result.phash] = imagehash.hex_to_hash(result.phash)
                            except (ValueError, TypeError):
                                pass
                        if candidate.thumbnail_url:
                            seen_urls.add(candidate.thumbnail_url)
                        existing_count += 1
                        next_file_index += 1
                        download_count += 1
                    except (HTTPError, URLError, TimeoutError, ValueError, OSError) as exc:
                        row["error"] = str(exc)
                    except Exception as exc:
                        row["error"] = f"unexpected error: {exc}"

                    rows_to_append.append(row)

        _append_manifest_rows(manifest, rows_to_append)
        print(
            f"[done] {member.member_id}: downloaded {download_count} "
            f"new images, filtered {filtered_count} candidates, "
            f"total files={existing_count}"
        )
        if delay_ms > 0:
            time.sleep(delay_ms / 1000)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crawl idol member images from DuckDuckGo image search.")
    parser.add_argument("--members", default="data/members.csv")
    parser.add_argument("--output-dir", default="data/raw_images")
    parser.add_argument("--manifest", default="data/raw_images/manifest.csv")
    parser.add_argument("--limit-per-member", type=int, default=15)
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=120,
        help="전체 쿼리 합산으로 수집할 후보 수 (각 쿼리엔 max_candidates/len(queries) 배분)",
    )
    parser.add_argument("--timeout-seconds", type=int, default=20)
    parser.add_argument("--min-bytes", type=int, default=10_000)
    parser.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)
    parser.add_argument("--min-side", type=int, default=400, help="짧은 변이 이 값 미만인 이미지 거부.")
    parser.add_argument("--min-blur", type=float, default=80.0, help="Laplacian variance 하한 (0 이면 비활성).")
    parser.add_argument("--delay-ms", type=int, default=1000)
    parser.add_argument("--download-workers", type=int, default=4)
    parser.add_argument("--phash-threshold", type=int, default=PHASH_HAMMING_THRESHOLD)
    parser.add_argument(
        "--no-dedup-across-members",
        action="store_true",
        help="다른 멤버 폴더에 이미 있는 동일 이미지(phash 기준)도 받는다.",
    )
    parser.add_argument(
        "--query-template",
        action="append",
        default=None,
        help=(
            "쿼리 템플릿을 덮어쓴다. {hint}, {group_en}, {member_en} 치환됨. "
            "여러 번 쓸 수 있다. 생략하면 기본 7개(한국어 고화질/직캠/셀카/포토 + 영문 photoshoot/fancam) 사용."
        ),
    )
    parser.add_argument("--member-ids", nargs="*")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    members = load_members(args.members)
    if args.member_ids:
        allowed = {member_id.strip() for member_id in args.member_ids if member_id.strip()}
        members = [member for member in members if member.member_id in allowed]

    if not members:
        raise SystemExit("No members available to crawl.")

    templates = tuple(args.query_template) if args.query_template else DEFAULT_QUERY_TEMPLATES

    crawl_members(
        members=members,
        output_dir=args.output_dir,
        manifest_path=args.manifest,
        limit_per_member=args.limit_per_member,
        max_candidates=args.max_candidates,
        timeout_seconds=args.timeout_seconds,
        min_bytes=args.min_bytes,
        max_bytes=args.max_bytes,
        min_side=args.min_side,
        min_blur=args.min_blur,
        delay_ms=args.delay_ms,
        download_workers=max(1, args.download_workers),
        query_templates=templates,
        phash_threshold=args.phash_threshold,
        dedup_across_members=not args.no_dedup_across_members,
    )


if __name__ == "__main__":
    main()
