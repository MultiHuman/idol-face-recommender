# Idol Face Recommender

로컬에서 아이돌 멤버 사진을 모으고, 내가 좋아하는 멤버 몇 명을 입력하면 취향에 맞을 것 같은 다른 멤버를 순위로 추천하는 프로젝트다.

지금 단계의 목표는 `완벽한 취향 AI`가 아니라, `얼굴 임베딩 기반 MVP`를 빠르게 만드는 것이다.

## Streamlit Community Cloud 배포

추천 런타임은 **member_vectors.csv / member_vectors_farl.csv / members.csv / group_genders.csv** 만 있으면 돌아간다. 원본 이미지·FaRL weights·insightface 모델은 `.gitignore` 로 제외되어 배포되지 않는다.

**배포 전 체크리스트:**
1. `requirements.txt` — 배포용 최소본 (streamlit, pandas, numpy)
2. `requirements-pipeline.txt` — 크롤/얼굴 추출/FaRL 같이 로컬 파이프라인 전용 의존성
3. 앱 진입점: `src/app.py`

**배포 절차:**
1. GitHub repo 생성 후 이 저장소 push
2. <https://share.streamlit.io> 에서 **New app** → 방금 만든 repo 선택
3. **Main file path** 란에 `src/app.py` 입력
4. **Python version** 을 3.11 이상으로 지정
5. Deploy 누르면 자동 빌드 → 공개 URL 발급

**로컬 테스트:**
```bash
pip install -r requirements.txt
streamlit run src/app.py
```

## 파이프라인 (크롤/추출/집계) 로컬 실행용

```bash
conda env create -f environment.yml
conda activate idol-face-rec
pip install -r requirements-pipeline.txt
```

## MVP 정의

- 멤버 목록 CSV를 만든다.
- 멤버별 사진을 수집한다.
- 사진에서 얼굴을 검출하고 crop 한다.
- 얼굴 임베딩을 추출한다.
- 멤버별 평균 벡터를 만든다.
- 좋아하는 멤버들의 평균 벡터와 가장 가까운 멤버를 추천한다.

추천 로직은 아래 순서로 고정한다.

1. 얼굴 이미지마다 임베딩 벡터를 추출한다.
2. 같은 멤버의 벡터를 평균 내서 `member_vector`를 만든다.
3. 사용자가 고른 최애들의 `member_vector`를 평균 내서 `taste_vector`를 만든다.
4. `taste_vector`와 모든 멤버 벡터의 코사인 유사도를 계산한다.
5. 입력에 사용한 멤버는 제외하고 Top N을 보여준다.

## 현재 포함된 것

- MVP 명세 문서: `docs/mvp.md`
- 이미지 크롤러: `src/crawl.py`
- 얼굴 추출 및 임베딩 생성기: `src/extract_faces.py`
- 멤버 추천 엔진: `src/recommend.py`
- 이미지 임베딩을 멤버 벡터로 집계하는 스크립트: `src/build_member_vectors.py`
- 간단한 Streamlit 앱: `src/app.py`
- 데이터 템플릿 CSV
- 기본 `data/members.csv`: 8개 그룹, 45명 1차 MVP 배치

## 폴더 구조

```text
.
├─ data/
│  ├─ members.template.csv
│  ├─ image_embeddings.template.csv
│  ├─ member_vectors.template.csv
│  ├─ raw_images/
│  ├─ face_crops/
│  └─ embeddings/
├─ docs/
│  └─ mvp.md
├─ src/
│  ├─ __init__.py
│  ├─ app.py
│  ├─ build_member_vectors.py
│  ├─ crawl.py
│  ├─ extract_faces.py
│  └─ recommend.py
└─ requirements.txt
```

## 데이터 형식

### `data/members.template.csv`

아이돌 멤버 마스터 목록이다.

- `member_id`
- `group_name`
- `member_name`
- `search_hint`
- `include_terms` (optional)
- `exclude_terms` (optional)

`include_terms`와 `exclude_terms`는 `|`로 구분해서 쓴다.  
이 값들은 크롤러가 검색 쿼리를 더 구체화하고, Bing 결과 메타데이터의 제목/설명/출처 URL을 보고 잘못된 후보를 걸러낼 때 사용된다.

### `data/image_embeddings.template.csv`

얼굴 검출과 임베딩 추출이 끝난 뒤 쌓이는 중간 결과다.

- `member_id`
- `group_name`
- `member_name`
- `image_path`
- `crop_path`
- `vector_json`
- `is_valid_face`
- `quality_score`
- `det_score`
- `face_area_ratio`
- `face_bbox_json`
- `embedding_dim`
- `error`

### `data/member_vectors.template.csv`

추천 엔진이 바로 읽는 최종 요약 파일이다.

- `member_id`
- `group_name`
- `member_name`
- `image_count`
- `vector_json`

## 시작 방법

```bash
conda env create -f environment.yml
conda activate idol-face-rec
```

이미 새 Conda 환경을 만들어둔 상태라면 그 환경을 활성화한 뒤 아래처럼 설치해도 된다.

```bash
pip install -r requirements.txt
```

크롤러는 Selenium을 사용한다.

Edge 또는 Chrome이 설치되어 있으면 Selenium이 맞는 드라이버를 자동으로 잡아 실행한다.
Windows에서는 기본 브라우저로 `msedge`를 먼저 사용한다.

멤버 CSV가 아직 없다면 템플릿을 복사해서 시작한다.

```bash
copy data\members.template.csv data\members.csv
```

이미지 크롤링 (쿼리 rotation + pHash/해상도/blur 필터 기본 적용):

```bash
python -m src.crawl --members data/members.csv --limit-per-member 15
```

기본으로 7개 쿼리 (한국어 고화질/직캠/셀카/포토 + 영문 photoshoot/fancam)가 회전된다.
각 쿼리엔 `max_candidates/쿼리수` 만큼 배분되며, pHash 해밍거리 ≤ 6 이면 멤버 내
near-duplicate 로 취급되어 버려진다. 짧은 변 400px 미만 / Laplacian blur variance 80 미만도 거부.

YouTube 팬캠에서 키프레임을 뽑고 싶으면 (데이터 다양성 최강):

```bash
python -m src.collect_fancams --members data/members.csv --limit-per-member 30 \
  --videos-per-member 3 --frames-per-video 12 --member-ids aespa__karina
```

얼굴 추출과 임베딩 생성 (antelopev2 사용 시 품질 ↑, 속도 ↓):

```bash
python -m src.extract_faces --members data/members.csv --overwrite
# 더 정확한 r100 임베딩:
python -m src.extract_faces --members data/members.csv --model-name antelopev2 --overwrite
# 성별/나이도 같이 저장:
python -m src.extract_faces --members data/members.csv --enable-genderage
```

샘플만 빠르게 테스트:

```bash
python -m src.extract_faces --members data/members.csv --max-images 4 --overwrite
```

추천 엔진 CLI (z-score 융합 + MMR 다양성 + 그룹 캡):

```bash
# 기본 (ArcFace 단일):
python -m src.recommend --vectors data/member_vectors.csv \
  --like newjeans__minji ive__wonyoung --top-k 10

# ArcFace + FaRL 융합 + 그룹당 최대 2명 + MMR λ=0.8:
python -m src.recommend \
  --vectors data/member_vectors.csv \
  --vectors-secondary data/member_vectors_farl.csv \
  --weights 1.0 0.8 \
  --mmr-lambda 0.8 --max-per-group 2 \
  --like newjeans__minji ive__wonyoung --top-k 10
```

멤버 벡터 집계 (Karcher spherical mean + IQR outlier trimming):

```bash
python -m src.build_member_vectors --input data/image_embeddings.csv \
  --output data/member_vectors.csv --min-images 5 --trim-fraction 0.15
```

융합 가중치 자동 튜닝 (LOO top-1):

```bash
python -m src.tune_fusion --min-images 5 --max-members 100
```

웹 앱:

```bash
streamlit run src/app.py
```

## 권장 개발 순서

1. `data/members.csv`를 실제 멤버 목록으로 채운다.
2. `src/crawl.py`로 멤버당 10~30장 정도의 이미지를 수집한다.
3. `src.extract_faces.py`로 얼굴 crop 과 임베딩을 추출한다.
4. `data/image_embeddings.csv`를 생성한다.
5. `src/build_member_vectors.py`로 `data/member_vectors.csv`를 만든다.
6. `src/app.py`에서 추천 결과를 확인한다.

## 현재 개선 사항 요약 (2026-04)

### 크롤링
- 7개 쿼리 회전 (직캠/셀카/고화질/포토/photoshoot/fancam)
- pHash (ImageHash) near-duplicate 제거 (멤버 내 + 크로스 멤버)
- 해상도 / Laplacian blur variance 필터
- source_page 도메인 기반 동적 Referer
- manifest 에 phash/width/height/blur_score 기록

### 얼굴 추출
- `antelopev2` (SCRFD + ArcFace r100) 옵션 추가
- genderage 모듈 enable 옵션 + pose yaw/pitch/roll 기록

### 임베딩 집계
- spherical (Karcher) mean 기반 centroid
- centroid 유사도 하위 15% IQR trimming
- intra-member pairwise cosine 을 confidence 컬럼으로 기록

### 추천
- spherical mean taste vector
- ArcFace + FaRL z-score 융합 (배경 분포로 정규화)
- confidence 기반 점수 스케일링
- MMR 재순위 + 그룹당 최대 N명 제약
- Streamlit 앱에서 MMR/그룹캡/가중치 실시간 조정

### 평가
- `src.tune_fusion`: Leave-one-out top-1 로 ArcFace/FaRL 가중치 그리드 서치

## 다음에 붙일 만한 것

- Naver API (등록 후 `src/crawl_naver.py` 구현)
- learned linear ranker (features → logistic regression, LOO objective 위에)
- 2-hop CF 확장 추천
