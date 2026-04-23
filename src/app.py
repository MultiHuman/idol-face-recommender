from __future__ import annotations

import csv
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


def main() -> None:
    st.set_page_config(page_title="아이돌 얼굴 추천", page_icon=":sparkles:", layout="wide")
    st.title("아이돌 얼굴 추천")
    st.caption("좋아하는 멤버를 검색하면 얼굴이 비슷한 아이돌을 추천해준다.")

    arcface_members = _load_members_arcface()
    farl_members = _load_members_farl()

    if not arcface_members:
        st.warning("`data/member_vectors.csv`가 아직 없다. 템플릿을 참고해서 데이터를 만든 뒤 다시 실행해줘.")
        st.code(
            "python -m src.build_member_vectors --input data/image_embeddings.csv --output data/member_vectors.csv",
            language="bash",
        )
        return

    # 추천 방식 선택
    mode_options = ["동일 인물 느낌 (ArcFace)"]
    if farl_members:
        mode_options.extend(["비슷한 분위기 (FaRL)", "혼합 (ArcFace + FaRL z-score 융합)"])
    engine = st.radio("추천 방식", mode_options, horizontal=True)

    if engine.startswith("혼합") and farl_members:
        primary = arcface_members
        secondary = farl_members
    elif engine.startswith("비슷한") and farl_members:
        primary = farl_members
        secondary = None
    else:
        primary = arcface_members
        secondary = None

    ko_labels = _load_korean_labels(DEFAULT_MEMBERS_PATH)

    def _label_for(member_id: str, fallback_group: str = "", fallback_name: str = "") -> str:
        if member_id in ko_labels:
            return ko_labels[member_id]
        if fallback_group or fallback_name:
            return f"{fallback_group} {fallback_name}".strip()
        return member_id

    labels = {
        _label_for(member.member_id, member.group_name, member.member_name): member.member_id
        for member in primary
    }
    sorted_label_keys = sorted(labels.keys())

    mode = st.radio("모드 선택", ["1명 검색", "여러 명 조합"], horizontal=True)

    if mode == "1명 검색":
        selected_label = st.selectbox(
            "좋아하는 멤버를 검색해봐",
            options=[None] + sorted_label_keys,
            format_func=lambda x: "멤버 이름이나 그룹명을 입력해봐" if x is None else x,
        )
        selected_ids = [labels[selected_label]] if selected_label else []
    else:
        selected_labels = st.multiselect(
            "좋아하는 멤버를 여러 명 골라봐",
            options=sorted_label_keys,
            placeholder="멤버를 선택하면 추천이 계산된다.",
        )
        selected_ids = [labels[label] for label in selected_labels]

    with st.expander("고급 옵션", expanded=False):
        top_k = st.slider("추천 인원 수", min_value=3, max_value=20, value=10)
        mmr_lambda = st.slider(
            "MMR 균형 (1.0 = 유사도 순, 0.5 = 다양성 강조)",
            min_value=0.3, max_value=1.0, value=0.8, step=0.05,
        )
        max_per_group = st.slider(
            "같은 그룹 최대 N명 (0 = 무제한)",
            min_value=0, max_value=5, value=2, step=1,
        )
        arcface_weight = st.slider("ArcFace 가중치", 0.0, 2.0, 1.0, 0.1)
        farl_weight = st.slider("FaRL 가중치", 0.0, 2.0, 1.0, 0.1) if secondary else 0.0
        gender_mode = st.radio(
            "성별 필터",
            options=["auto", "off", "F", "M"],
            index=0,
            horizontal=True,
            help="auto = 좋아하는 멤버 다수 성별만, off = 필터 없음, F/M = 명시 강제.",
        )

    if not selected_ids:
        st.info("멤버를 선택하면 비슷한 얼굴의 아이돌이 추천된다.")
        return

    weights = (arcface_weight, farl_weight) if secondary else (1.0,)
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
        st.error("추천 결과를 만들지 못했다. 데이터 차원이나 입력 멤버를 확인해줘.")
        return

    st.subheader("추천 결과")
    display_rows = []
    for index, row in enumerate(rows, start=1):
        mid = str(row.get("member_id", ""))
        display_rows.append(
            {
                "순위": index,
                "멤버": _label_for(mid, str(row.get("group_name", "")), str(row.get("member_name", ""))),
                "점수(z)": round(float(row.get("score", 0.0)), 3),
                "신뢰도": round(float(row.get("confidence", 0.0)), 2),
                "사진": int(row.get("image_count", 0)),
            }
        )
    dataframe = pd.DataFrame(display_rows)
    st.dataframe(dataframe, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
