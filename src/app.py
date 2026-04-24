from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

# Streamlit Cloud 는 src/app.py 를 직접 실행해서 sys.path[0] 가 src/ 가 된다.
# repo 루트를 path 앞에 넣어야 `from src.recommend import ...` 가 해결됨.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd
import streamlit as st

from src.recommend import load_member_vectors, recommend_from_members


ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_VECTORS_PATH = ROOT_DIR / "data" / "member_vectors.csv"
FARL_VECTORS_PATH = ROOT_DIR / "data" / "member_vectors_farl.csv"
DEFAULT_MEMBERS_PATH = ROOT_DIR / "data" / "members.csv"


@st.cache_data
def _load_members_arcface():
    return load_member_vectors(DEFAULT_VECTORS_PATH)


@st.cache_data
def _load_members_farl():
    if FARL_VECTORS_PATH.exists():
        return load_member_vectors(FARL_VECTORS_PATH)
    return []


def _clean_label(text: str) -> str:
    """'"뉴진스" 민지' 처럼 쿼리용 따옴표가 섞인 표기를 '뉴진스 민지' 로 다듬는다."""
    if not text:
        return text
    cleaned = re.sub(r'["“”]', "", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _load_korean_labels(members_csv: Path) -> dict[str, str]:
    labels: dict[str, str] = {}
    if not members_csv.exists():
        return labels
    with members_csv.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            mid = (row.get("member_id") or "").strip()
            hint = _clean_label(row.get("search_hint") or "")
            if mid and hint:
                labels[mid] = hint
    return labels


def main() -> None:
    st.set_page_config(page_title="아이돌 얼굴 추천기", page_icon="✨", layout="wide")
    st.title("아이돌 얼굴 추천기")
    st.caption("좋아하는 멤버를 고르면 비슷한 느낌의 다른 아이돌을 찾아드립니다.")

    arcface_members = _load_members_arcface()
    farl_members = _load_members_farl()

    if not arcface_members:
        st.warning("추천에 필요한 데이터 파일이 아직 준비되지 않았어요.")
        return

    ko_labels = _load_korean_labels(DEFAULT_MEMBERS_PATH)

    def _label_for(member_id: str, fallback_group: str = "", fallback_name: str = "") -> str:
        """한글 힌트와 영문 그룹·멤버명을 함께 담아서 어느 쪽 키워드로도 검색되게 한다."""
        ko = ko_labels.get(member_id, "").strip()
        en = _clean_label(f"{fallback_group} {fallback_name}".strip())
        if ko and en and ko.lower() != en.lower():
            return f"{ko} · {en}"
        return ko or en or member_id

    labels = {
        _label_for(member.member_id, member.group_name, member.member_name): member.member_id
        for member in arcface_members
    }
    sorted_labels = sorted(labels.keys(), key=lambda s: s.lower())

    selected_labels = st.multiselect(
        "좋아하는 멤버 (여러 명 선택 가능)",
        options=sorted_labels,
        placeholder="그룹명이나 멤버 이름을 입력해 보세요",
    )
    selected_ids = [labels[lbl] for lbl in selected_labels]

    with st.expander("설정 바꾸기", expanded=False):
        engine_options = ["얼굴 + 분위기 섞어서 (추천)"]
        if farl_members:
            engine_options.append("얼굴만 (닮은 사람)")
            engine_options.append("분위기만 (이미지 톤)")
        else:
            engine_options = ["얼굴만 (닮은 사람)"]

        engine_label = st.radio(
            "추천 기준",
            options=engine_options,
            index=0,
            help=(
                "· 얼굴 + 분위기: 얼굴 생김새와 분위기를 모두 고려 (기본)\n"
                "· 얼굴만: 눈·코·입 배치 등 골격이 닮은 사람 위주\n"
                "· 분위기만: 이미지 톤·스타일이 비슷한 사람 위주"
            ),
        )

        top_k = st.slider("추천할 인원 수", min_value=3, max_value=20, value=10)
        max_per_group = st.slider(
            "같은 그룹에서 최대 몇 명",
            min_value=0, max_value=5, value=2,
            help="0이면 제한 없음. 기본 2명으로 다양성 확보.",
        )
        mmr_lambda = st.slider(
            "닮음 ↔ 다양성",
            min_value=0.3, max_value=1.0, value=0.8, step=0.05,
            help="오른쪽일수록 닮은 순, 왼쪽일수록 여러 인상이 섞임",
        )

        gender_display = st.radio(
            "성별 제한",
            options=["자동 (고른 사람과 같은 성별)", "제한 없음", "여성만", "남성만"],
            index=0,
            horizontal=True,
        )

    # 설정 → 내부 옵션 변환
    gender_map = {
        "자동 (고른 사람과 같은 성별)": "auto",
        "제한 없음": "off",
        "여성만": "F",
        "남성만": "M",
    }
    gender_mode = gender_map[gender_display]

    if engine_label.startswith("얼굴 + 분위기"):
        primary = arcface_members
        secondary = farl_members if farl_members else None
        weights = (1.0, 0.5) if farl_members else (1.0,)
    elif engine_label.startswith("얼굴만"):
        primary = arcface_members
        secondary = None
        weights = (1.0,)
    else:
        primary = farl_members if farl_members else arcface_members
        secondary = None
        weights = (1.0,)

    if not selected_ids:
        st.info("👈 멤버를 고르면 비슷한 아이돌이 여기에 나타납니다.")
        return

    rows = recommend_from_members(
        primary,
        liked_member_ids=selected_ids,
        top_k=top_k,
        secondary_members=secondary,
        weights=weights,
        mmr_lambda=mmr_lambda,
        max_per_group=max_per_group,
        pool_size=max(50, top_k * 5),
        gender_filter=gender_mode,
    )

    if not rows:
        st.error("조건에 맞는 추천 결과가 없어요. 설정을 바꾸거나 다른 멤버를 골라 보세요.")
        return

    st.subheader("추천 결과")
    display_rows = []
    for index, row in enumerate(rows, start=1):
        mid = str(row.get("member_id", ""))
        display_rows.append(
            {
                "순위": index,
                "멤버": _label_for(mid, str(row.get("group_name", "")), str(row.get("member_name", ""))),
                "유사도": round(float(row.get("score", 0.0)), 3),
                "사진 수": int(row.get("image_count", 0)),
            }
        )
    dataframe = pd.DataFrame(display_rows)
    st.dataframe(dataframe, use_container_width=True, hide_index=True)

    st.caption(
        "유사도는 상대값 (z-score)이에요. 양수일수록 선택한 멤버들과 더 비슷한 쪽이고, "
        "그룹/성별 필터로 걸러낸 뒤 다양성까지 고려해서 순위를 매깁니다."
    )


if __name__ == "__main__":
    main()
