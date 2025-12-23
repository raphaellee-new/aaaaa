# -*- coding: utf-8 -*-
# Auto-generated from Simple_Agent.ipynb (code cells only)
# NOTE: Review and remove any Colab-only imports/usages if present (google.colab.*).

# ------------------------------
# Cell 1
# ------------------------------
SERVICE_KEY = "FqDrDXAFVLM5UoSYzzyC619C0HzxHTDx7a3Ot5Q0M90m62wl700+7jt0PTBpFhP1qivmuO5jVLgmiJHuPlPkNA=="
BASE_URL = "https://apis.data.go.kr/1371000/policyNewsService/policyNewsList"

# ------------------------------
# Cell 2
# ------------------------------
# ============================================================
# Korea.kr 정책뉴스(정책브리핑) 수집 -> DataFrame -> CSV 저장/다운로드
# Google Colab "한 셀" 실행용
# ============================================================


import re
import requests
import pandas as pd
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
from urllib.parse import quote
from datetime import datetime, timedelta
from google.colab import files

API_URL = "https://apis.data.go.kr/1371000/policyNewsService/policyNewsList"

# ----------------------------
# 0) 유틸
# ----------------------------
def normalize_service_key(service_key: str) -> str:
    """
    - data.go.kr에서 serviceKey가 'Encoding(%)' 형태면 그대로 사용
    - 'Decoding(+ / = 포함)' 형태면 URL-safe로 인코딩
    """
    k = service_key.strip()
    return k if "%" in k else quote(k, safe="")

def mask_url(u: str) -> str:
    return re.sub(r"(serviceKey=)[^&]+", r"\1***", u)

def strip_html(html: str) -> str:
    if not html:
        return ""
    soup = BeautifulSoup(html, "lxml")
    text = soup.get_text("\n")
    return re.sub(r"\n{3,}", "\n\n", text).strip()

def strip_namespaces(root: ET.Element) -> ET.Element:
    """
    ElementTree 태그에 {namespace} 가 붙는 경우 제거
    """
    for el in root.iter():
        if isinstance(el.tag, str) and el.tag.startswith("{"):
            el.tag = el.tag.split("}", 1)[1]
    return root

def daterange_chunks(start_yyyymmdd: str, end_yyyymmdd: str, chunk_days: int = 3):
    """
    API가 3일 초과 조회 시 THREE_DAYS_OVER_ERROR를 반환하므로,
    inclusive 날짜 구간을 최대 3일 단위로 쪼개 호출한다.
    """
    start = datetime.strptime(start_yyyymmdd, "%Y%m%d").date()
    end   = datetime.strptime(end_yyyymmdd, "%Y%m%d").date()
    cur = start
    while cur <= end:
        chunk_end = min(cur + timedelta(days=chunk_days - 1), end)
        yield cur.strftime("%Y%m%d"), chunk_end.strftime("%Y%m%d")
        cur = chunk_end + timedelta(days=1)

# ----------------------------
# 1) API 호출 + 파싱
# ----------------------------
def parse_items_from_xml(xml_text: str, debug: bool = False):
    root = ET.fromstring(xml_text)

    # resultCode/resultMsg
    result_code = (root.findtext(".//resultCode") or "").strip()
    result_msg  = (root.findtext(".//resultMsg")  or "").strip()

    # namespace 제거 후 item 탐색
    root = strip_namespaces(root)

    items = root.findall(".//item")
    # 혹시 item 태그명이 다르게 끝나는 경우 대비(희귀 케이스)
    if len(items) == 0:
        items = [el for el in root.iter() if isinstance(el.tag, str) and el.tag.lower().endswith("item")]

    if debug:
        print("  [PARSE] resultCode:", result_code, "resultMsg:", result_msg, "items:", len(items))

    rows = []
    for it in items:
        def t(tag):
            x = it.find(tag)
            return x.text.strip() if (x is not None and x.text) else ""

        contents_type = t("ContentsType")
        data_contents = t("DataContents")

        rows.append({
            "NewsItemId": t("NewsItemId"),
            "GroupingCode": t("GroupingCode"),
            "MinisterCode": t("MinisterCode"),
            "ApproveDate": t("ApproveDate"),
            "ModifyDate": t("ModifyDate"),
            "Title": t("Title"),
            "SubTitle1": t("SubTitle1"),
            "SubTitle2": t("SubTitle2"),
            "SubTitle3": t("SubTitle3"),
            "ContentsType": contents_type,
            "OriginalUrl": t("OriginalUrl"),
            "ThumbnailUrl": t("ThumbnailUrl"),
            "OriginalimgUrl": t("OriginalimgUrl"),
            "DataContents_text": strip_html(data_contents) if contents_type.upper() == "H" else data_contents,
            "DataContents_raw": data_contents,
        })

    meta = {"resultCode": result_code, "resultMsg": result_msg}
    return rows, meta

def call_api(service_key: str, start_date: str, end_date: str, timeout: int = 30, debug: bool = True):
    sk = normalize_service_key(service_key)
    url = f"{API_URL}?serviceKey={sk}&startDate={start_date}&endDate={end_date}"

    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()

    if debug:
        print("[CALL]", start_date, "~", end_date,
              "|", resp.status_code,
              "|", mask_url(resp.url),
              "| len:", len(resp.text))

    rows, meta = parse_items_from_xml(resp.text, debug=False)

    # resultCode가 0이 아니면 에러로 취급
    if meta.get("resultCode") != "0":
        if debug:
            print("  -> API ERROR:", meta.get("resultCode"), meta.get("resultMsg"))
            print("  -> response head:", resp.text[:250])
        return [], meta, resp.text

    return rows, meta, resp.text

def fetch_policy_news_range(service_key: str, start_date: str, end_date: str, debug: bool = True):
    all_rows = []
    errors = []

    for s, e in daterange_chunks(start_date, end_date, chunk_days=3):
        rows, meta, _ = call_api(service_key, s, e, debug=debug)
        if meta.get("resultCode") == "0":
            for r in rows:
                r["req_startDate"] = s
                r["req_endDate"] = e
            all_rows.extend(rows)
        else:
            errors.append({"startDate": s, "endDate": e, **meta})

    df = pd.DataFrame(all_rows)
    err_df = pd.DataFrame(errors)

    # 중복 제거
    if not df.empty and "NewsItemId" in df.columns:
        df = df.drop_duplicates(subset=["NewsItemId"]).reset_index(drop=True)

    # 콘텐츠 길이(요약/필터링용)
    if not df.empty and "DataContents_text" in df.columns:
        df["content_length"] = df["DataContents_text"].astype(str).str.len()

    return df, err_df

# ----------------------------
# 2) 실행 파라미터
# ----------------------------
# data.go.kr에서 발급받은 serviceKey 입력

# 날짜 범위: YYYYMMDD
# - 기본은 "최근 3일"로 설정(에러 없이 동작 확인 용이)
today = datetime.now().date()
START_DATE = (today - timedelta(days=2)).strftime("%Y%m%d")
END_DATE   = today.strftime("%Y%m%d")

# 사용자가 직접 지정하고 싶으면 위 두 줄 대신 아래처럼 바꾸세요:
# START_DATE = "20251215"
# END_DATE   = "20251222"

print("=== RUN PARAMS ===")
print("START_DATE:", START_DATE, "END_DATE:", END_DATE)

# ----------------------------
# 3) 수집 실행
# ----------------------------
df, err_df = fetch_policy_news_range(SERVICE_KEY, START_DATE, END_DATE, debug=True)

print("\n=== RESULT SUMMARY ===")
print("rows:", len(df))
print("errors:", len(err_df))
if not err_df.empty:
    display(err_df)

display(df.head(5) if not df.empty else pd.DataFrame([{"status": "no rows"}]))

# ----------------------------
# 4) CSV 저장 + 다운로드
# ----------------------------
# 컬럼 정렬(존재하는 것만)
ordered_cols = [
    "NewsItemId", "ApproveDate", "ModifyDate", "Title",
    "SubTitle1", "SubTitle2", "SubTitle3",
    "GroupingCode", "MinisterCode",
    "OriginalUrl", "ContentsType",
    "content_length", "DataContents_text",
    "req_startDate", "req_endDate",
]
if not df.empty:
    ordered_cols = [c for c in ordered_cols if c in df.columns]
    df_out = df[ordered_cols].copy()
else:
    df_out = df.copy()

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_path = f"policy_news_{START_DATE}_{END_DATE}_{timestamp}.csv"

df_out.to_csv(csv_path, index=False, encoding="utf-8-sig")
print("\nSaved CSV:", csv_path)

# Colab 다운로드
files.download(csv_path)

# ------------------------------
# Cell 3
# ------------------------------
import os
os.environ["MISTRAL_API_KEY"] = "sLWOHYqD1V63JCo5MvL3jbKJkEjGieO6"

# ------------------------------
# Cell 4
# ------------------------------
# =========================
# Google Colab One-Cell Script
# CSV (기본경로) -> Mistral AI -> 2줄 요약 + 키워드 -> 결과 CSV
# =========================


import os
import json
import time
import re
import pandas as pd
from tqdm import tqdm
from getpass import getpass

from mistralai import Mistral

# -------------------------------------------------
# 0. 파일 경로 및 컬럼 설정 (Colab 기본 경로)
# -------------------------------------------------
INPUT_CSV = "/content/policy_news_20251220_20251222_20251222_192215.csv"
OUTPUT_CSV = "/content/policy_news_with_summary_keywords.csv"

TEXT_COL = "DataContents_text"   # 본문
TITLE_COL = "Title"              # 제목
SUB1_COL = "SubTitle1"            # 부제 (없으면 자동 무시)

MODEL = "mistral-small-latest"
MAX_CONTENT_CHARS = 4500
SLEEP_SEC = 0.3
MAX_RETRIES = 5

# -------------------------------------------------
# 1. Mistral API Key 설정
# -------------------------------------------------
api_key = os.environ.get("MISTRAL_API_KEY")
if not api_key:
    api_key = getpass("MISTRAL_API_KEY 입력 (입력값 비표시): ").strip()

client = Mistral(api_key=api_key)

# -------------------------------------------------
# 2. 유틸 함수
# -------------------------------------------------
def truncate_text(text, max_chars):
    if pd.isna(text):
        return ""
    text = str(text)
    return text if len(text) <= max_chars else text[:max_chars] + "\n...(truncated)"

def safe_json_parse(text):
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                return None
    return None

def mistral_summarize(title, subtitle, content):
    prompt = f"""
[제목] {title}
[부제] {subtitle}

[본문]
{truncate_text(content, MAX_CONTENT_CHARS)}

요청:
- 한국어로 2줄 요약
- 핵심 키워드 5~10개
- 반드시 JSON 형식으로만 응답

형식:
{{
  "summary_line1": "...",
  "summary_line2": "...",
  "keywords": ["...", "..."]
}}
""".strip()

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.complete(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "You are a Korean policy/news analyst. Output JSON only."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )

            content = response.choices[0].message.content
            data = safe_json_parse(content)
            if not data:
                raise ValueError("JSON 파싱 실패")

            return {
                "summary_line1": data.get("summary_line1", ""),
                "summary_line2": data.get("summary_line2", ""),
                "keywords": ", ".join(dict.fromkeys(data.get("keywords", [])))
            }

        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                return {
                    "summary_line1": "",
                    "summary_line2": "",
                    "keywords": "",
                    "error": str(e)
                }
            time.sleep(2 ** attempt)

# -------------------------------------------------
# 3. CSV 로드
# -------------------------------------------------
df = pd.read_csv(INPUT_CSV)

if TEXT_COL not in df.columns:
    raise ValueError(f"본문 컬럼 '{TEXT_COL}' 이 존재하지 않습니다")

# -------------------------------------------------
# 4. 레코드별 요약 + 키워드 생성
# -------------------------------------------------
summary1, summary2, keywords, errors = [], [], [], []

for _, row in tqdm(df.iterrows(), total=len(df), desc="Mistral 처리 중"):
    result = mistral_summarize(
        row.get(TITLE_COL, ""),
        row.get(SUB1_COL, ""),
        row.get(TEXT_COL, "")
    )

    summary1.append(result.get("summary_line1", ""))
    summary2.append(result.get("summary_line2", ""))
    keywords.append(result.get("keywords", ""))
    errors.append(result.get("error", ""))

    time.sleep(SLEEP_SEC)

df["summary_line1"] = summary1
df["summary_line2"] = summary2
df["keywords"] = keywords
df["mistral_error"] = errors

# -------------------------------------------------
# 5. 결과 저장
# -------------------------------------------------
df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
print(f"완료: {OUTPUT_CSV}")

from google.colab import files
files.download(OUTPUT_CSV)

# ------------------------------
# Cell 5
# ------------------------------
# ------------------------------------------------------------
# (Google Colab 1-cell) policy_news_with_summary_keywords.csv
#  -> Mistral AI로 "보고서 초안(Markdown)" 생성 + 다운로드
#  - 권장 최종 구성(컬럼 고정 매핑: Title, DataContents_text, ApproveDate, OriginalUrl, summary_line1/2, keywords)
# ------------------------------------------------------------


import os
import re
import json
import pandas as pd
from datetime import datetime
from getpass import getpass

from mistralai import Mistral

# ========== 0) 입력 파일 자동 탐색 ==========
CANDIDATE_PATHS = [
    "policy_news_with_summary_keywords.csv",
    "/content/policy_news_with_summary_keywords.csv",
    "/mnt/data/policy_news_with_summary_keywords.csv",
]

INPUT_CSV = next((p for p in CANDIDATE_PATHS if os.path.exists(p)), None)
if not INPUT_CSV:
    raise FileNotFoundError(
        "CSV 파일을 찾지 못했습니다. Colab에 업로드했는지 확인하고, "
        "파일명이 다르면 CANDIDATE_PATHS에 추가하세요."
    )

# ========== 1) Mistral API Key 설정 ==========
# 권장: Colab Secrets/환경변수에 저장 (os.environ["MISTRAL_API_KEY"] = "...")
if not os.environ.get("MISTRAL_API_KEY"):
    os.environ["MISTRAL_API_KEY"] = getpass("Mistral API Key 입력 (표시되지 않음): ")

client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

# ========== 2) CSV 로드 ==========
df = pd.read_csv(INPUT_CSV)

# ========== 3) 권장 최종 컬럼 매핑(고정) ==========
required_cols = [
    "Title", "DataContents_text", "ApproveDate", "OriginalUrl",
    "summary_line1", "summary_line2", "keywords"
]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"필수 컬럼이 없습니다: {missing}\n현재 CSV 컬럼: {list(df.columns)}")

col_title = "Title"
col_body  = "DataContents_text"
col_date  = "ApproveDate"
col_url   = "OriginalUrl"
col_kw    = "keywords"

# 2줄 요약 결합
df["summary"] = (
    df["summary_line1"].fillna("").astype(str).str.strip()
    + "\n" +
    df["summary_line2"].fillna("").astype(str).str.strip()
)
col_sum = "summary"

# 결측/형 변환
for c in [col_title, col_body, col_date, col_url, col_kw, col_sum]:
    df[c] = df[c].fillna("").astype(str)

def clean_text(s: str) -> str:
    s = re.sub(r"\s+", " ", str(s)).strip()
    return s

# ========== 4) LLM 입력용 아이템 구성 ==========
MAX_BODY_CHARS_PER_ITEM = 1400   # 본문이 길면 토큰 초과 방지 위해 절단
MAX_ITEMS_FOR_ANALYSIS  = 80     # 데이터가 많으면 40~60으로 줄이는 것을 권장

items = []
for i, row in df.iterrows():
    items.append({
        "idx": i + 1,
        "title": clean_text(row[col_title]),
        "date": clean_text(row[col_date]),
        "url": clean_text(row[col_url]),
        "body": clean_text(row[col_body])[:MAX_BODY_CHARS_PER_ITEM],
        "summary": row[col_sum].strip(),
        "keywords": clean_text(row[col_kw]),
        "content_length": len(str(row[col_body])) if "content_length" in df.columns else None,
    })

analysis_items = items[:MAX_ITEMS_FOR_ANALYSIS]

analysis_payload = []
for it in analysis_items:
    analysis_payload.append({
        "idx": it["idx"],
        "title": it["title"],
        "date": it["date"],
        "url": it["url"],
        "summary": it["summary"],
        "keywords": it["keywords"],
        "body": it["body"],
    })

# ========== 5) Mistral 호출 유틸 ==========
MODEL = "mistral-small-latest"   # 필요 시 "mistral-medium-latest" 등으로 변경
TEMPERATURE = 0.2

def chat(messages, model=MODEL, temperature=TEMPERATURE):
    resp = client.chat.complete(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return resp.choices[0].message.content

def extract_json(text: str):
    text = (text or "").strip()
    m = re.search(r"(\{.*\})", text, flags=re.DOTALL)
    if not m:
        raise ValueError("모델 출력에서 JSON을 찾지 못했습니다.\n--- 모델 출력(앞부분) ---\n" + text[:1500])
    return json.loads(m.group(1))

# ========== 6) (1단계) 이슈 구조화 JSON 생성 ==========
system_1 = (
    "당신은 공공기관 보고서 작성 보조자입니다. "
    "아래 뉴스 목록을 바탕으로 핵심 이슈를 구조화해 주세요. "
    "출력은 반드시 JSON 하나만 반환하세요(코드블록/설명 금지)."
)

user_1 = f"""
[요청]
다음 뉴스 목록을 분석하여, 보고서 작성에 사용할 '이슈 구조화 결과'를 JSON으로 생성하세요.

[규칙]
- 반드시 JSON만 출력
- top_issues: 상위 5~8개 이슈
  - 각 이슈 객체 필드:
    - issue: 이슈명(짧게)
    - situation: 현황 요약(2~4문장)
    - why_it_matters: 왜 중요한가(공공기관 관점)
    - evidence_items: 근거 뉴스 idx 배열(2~8개)
    - risk_or_opportunity: 리스크/기회 요약
    - suggested_actions: 권고 조치 불릿 3~6개(문장형)
- 가능하면 keywords/summary를 우선 활용하고, 부족할 때 body를 참고

[뉴스 목록 JSON]
{json.dumps(analysis_payload, ensure_ascii=False)}
"""

issue_json_text = chat([
    {"role": "system", "content": system_1},
    {"role": "user", "content": user_1},
])

issue_struct = extract_json(issue_json_text)

# ========== 7) (2단계) 최종 보고서(Markdown) 생성 ==========
# 출처 목록 생성 (idx-title-date-url)
source_lines = []
for it in analysis_items:
    d = f" ({it['date']})" if it["date"] else ""
    u = f" | {it['url']}" if it["url"] else ""
    source_lines.append(f"- [{it['idx']}] {it['title']}{d}{u}")

system_2 = (
    "당신은 공공기관 보고서 작성자입니다. "
    "주어진 이슈 구조(JSON)와 출처 목록을 근거로 한국어 Markdown 보고서 초안을 작성하세요."
)

user_2 = f"""
[요청]
아래 이슈 구조(JSON)와 출처 목록을 기반으로 '보고서 초안'을 Markdown으로 작성하세요.

[보고서 형식]
# 정책뉴스 분석 보고서 초안 (기간/키워드는 상황에 맞게 작성)
## 목적
- 3~5줄

## 개요
- 불릿 5~8개 (핵심 이슈 요약)

## 본문
- 이슈별로 ### 소제목 구성
- 각 이슈마다 아래 소구조를 포함:
  - **현황 요약**: 2~4문장
  - **공공기관 시사점/리스크**: 불릿 2~4개
  - **권고 조치**: 불릿 2~5개
  - **근거 뉴스**: [idx] 형태로 2~6개 인용

## 출처
- 아래 출처 목록을 그대로 붙여 넣기

[이슈 구조 JSON]
{json.dumps(issue_struct, ensure_ascii=False)}

[출처 목록]
{chr(10).join(source_lines)}
"""

report_md = chat([
    {"role": "system", "content": system_2},
    {"role": "user", "content": user_2},
])

# ========== 8) 저장 + 다운로드 ==========
out_name = f"report_draft_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
with open(out_name, "w", encoding="utf-8") as f:
    f.write(report_md)

print("=== 보고서 초안 생성 완료 ===")
print("입력 CSV:", INPUT_CSV)
print("저장 파일:", out_name)

# Colab 다운로드
from google.colab import files
files.download(out_name)

# ------------------------------
# Cell 6
# ------------------------------
# =========================================================
# Google Colab 단일 셀
# Secrets(userdata.get) -> Markdown -> HTML -> Gmail 발송
# =========================================================

# 1. 필요한 패키지 설치

# 2. 라이브러리 import
import os
import re
import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import markdown as md
from google.colab import userdata

# =========================================================
# 3. Colab Secrets 로드 (userdata.get 방식)
# =========================================================
GMAIL_ADDRESS = userdata.get("GMAIL_ADDRESS")
TO_EMAIL = userdata.get("TO_EMAIL")

# 정상 키 + 오타 키 fallback
GMAIL_APP_PASSWORD = (
    userdata.get("GMAIL_APP_PASSWORD")
    or userdata.get("GMAIL_APP_PASSWORR")
)

missing = []
if not GMAIL_ADDRESS:
    missing.append("GMAIL_ADDRESS")
if not TO_EMAIL:
    missing.append("TO_EMAIL")
if not GMAIL_APP_PASSWORD:
    missing.append("GMAIL_APP_PASSWORD (또는 GMAIL_APP_PASSWORR)")

if missing:
    raise RuntimeError(
        f"Colab 보안 비밀(Secrets)에 값이 없습니다: {missing}\n"
        f"🔒 좌측 Secrets 메뉴에서 추가/수정 후 런타임 재시작하세요."
    )

print("✔ Secrets 로딩 완료")
print("  - From:", GMAIL_ADDRESS)
print("  - To  :", TO_EMAIL)
print("  - App Password: OK")

# =========================================================
# 4. Markdown 파일 경로 (Colab 기본 경로)
# =========================================================
MD_PATH = "/content/report_draft_20251222_193913.md"

if not os.path.exists(MD_PATH):
    md_files = [f for f in os.listdir("/content") if f.lower().endswith(".md")]
    raise FileNotFoundError(
        f"Markdown 파일을 찾을 수 없습니다: {MD_PATH}\n"
        f"/content 에 존재하는 md 파일: {md_files}"
    )

# =========================================================
# 5. Markdown 파일 읽기
# =========================================================
with open(MD_PATH, "r", encoding="utf-8") as f:
    md_text = f.read()

# =========================================================
# 6. Markdown -> HTML 변환 (포맷 유지)
# =========================================================
html_body = md.markdown(
    md_text,
    extensions=[
        "extra",
        "tables",
        "fenced_code",
        "sane_lists",
        "toc"
    ]
)

html_template = f"""
<!doctype html>
<html>
<head>
<meta charset="utf-8">
</head>
<body style="font-family: Arial, Helvetica, sans-serif; line-height:1.6; color:#111;">
  <div style="max-width:900px; margin:0 auto; padding:8px 4px;">
    {html_body}
    <hr style="margin-top:24px; border:none; border-top:1px solid #ddd;">
    <div style="font-size:12px; color:#666;">
      <div>발송 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
      <div>원본 파일: {MD_PATH}</div>
    </div>
  </div>
</body>
</html>
"""

plain_text = re.sub(r"\n{3,}", "\n\n", md_text).strip()

# =========================================================
# 7. 메일 MIME 구성
# =========================================================
SUBJECT = f"정책뉴스 분석 보고서 초안 ({datetime.now().strftime('%Y-%m-%d')})"

msg = MIMEMultipart("alternative")
msg["Subject"] = SUBJECT
msg["From"] = GMAIL_ADDRESS
msg["To"] = TO_EMAIL

msg.attach(MIMEText(plain_text, "plain", "utf-8"))
msg.attach(MIMEText(html_template, "html", "utf-8"))

# =========================================================
# 8. Gmail SMTP 발송
# =========================================================
def send_gmail(from_addr, app_password, to_addrs, mime_msg):
    recipients = [x.strip() for x in re.split(r"[;,]", to_addrs) if x.strip()]
    if not recipients:
        raise ValueError("수신자 주소가 비어 있습니다.")

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(from_addr, app_password)
        server.sendmail(from_addr, recipients, mime_msg.as_string())

send_gmail(GMAIL_ADDRESS, GMAIL_APP_PASSWORD, TO_EMAIL, msg)

print("✅ 메일 발송 완료")
print("  - Subject:", SUBJECT)

# ------------------------------
# Cell 7
# ------------------------------
# ============================================================
# DataContents_text → QA Retrieval Embedding (Google Colab)
# ============================================================

# 0. 라이브러리 설치 (최초 1회)

# ------------------------------------------------------------
# 1. 라이브러리 로드
# ------------------------------------------------------------
import pandas as pd
import numpy as np
import json
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# ------------------------------------------------------------
# 2. 설정
# ------------------------------------------------------------
INPUT_CSV = "/content/policy_news_with_summary_keywords.csv"

OUTPUT_META_CSV = "/content/policy_news_chunks_meta.csv"
OUTPUT_EMB_NPY  = "/content/policy_news_chunks_embeddings.npy"
OUTPUT_FAISS    = "/content/policy_news_faiss.index"

TEXT_COLUMN = "DataContents_text"
TITLE_COLUMN = "Title"  # 없으면 자동 무시됨

# 질문-답변 검색에 적합한 다국어 모델
MODEL_NAME = "intfloat/multilingual-e5-base"

# Chunk 설정 (문단 단위 검색 품질 향상)
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50

# ------------------------------------------------------------
# 3. 텍스트 분할 함수
# ------------------------------------------------------------
def chunk_text(text, chunk_size=300, overlap=50):
    chunks = []
    start = 0
    text = text.strip()
    length = len(text)

    while start < length:
        end = start + chunk_size
        chunk = text[start:end].strip()
        if len(chunk) > 30:
            chunks.append(chunk)
        start = end - overlap

    return chunks

# ------------------------------------------------------------
# 4. CSV 로드
# ------------------------------------------------------------
df = pd.read_csv(INPUT_CSV)

if TEXT_COLUMN not in df.columns:
    raise ValueError(
        f"'{TEXT_COLUMN}' 컬럼이 없습니다. 현재 컬럼: {list(df.columns)}"
    )

# ------------------------------------------------------------
# 5. 문서 → Chunk + Metadata 생성
# ------------------------------------------------------------
rows = []
chunk_id = 0

for doc_id, row in tqdm(df.iterrows(), total=len(df)):
    body = str(row[TEXT_COLUMN])
    title = str(row[TITLE_COLUMN]) if TITLE_COLUMN in df.columns else ""

    for idx, chunk in enumerate(chunk_text(body)):
        rows.append({
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "chunk_index": idx,
            "title": title,
            "text": chunk
        })
        chunk_id += 1

chunk_df = pd.DataFrame(rows)
print(f"[INFO] 생성된 Chunk 수: {len(chunk_df)}")

# ------------------------------------------------------------
# 6. 임베딩 모델 로드
# ------------------------------------------------------------
model = SentenceTransformer(MODEL_NAME)

# e5 모델 규칙: passage / query prefix 필수
texts_for_embedding = [
    f"passage: {t}"
    for t in chunk_df["text"].tolist()
]

# ------------------------------------------------------------
# 7. Chunk → Vector 변환
# ------------------------------------------------------------
embeddings = model.encode(
    texts_for_embedding,
    batch_size=32,
    show_progress_bar=True,
    normalize_embeddings=True
)

print(f"[INFO] Embedding shape: {embeddings.shape}")

# ------------------------------------------------------------
# 8. 저장
# ------------------------------------------------------------

# (1) 메타데이터 CSV
chunk_df.to_csv(OUTPUT_META_CSV, index=False, encoding="utf-8-sig")

# (2) 벡터 NPY
np.save(OUTPUT_EMB_NPY, embeddings)

# (3) FAISS 인덱스
dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)  # cosine similarity
index.add(embeddings)
faiss.write_index(index, OUTPUT_FAISS)

print("[DONE]")
print(f"- META CSV : {OUTPUT_META_CSV}")
print(f"- EMB NPY  : {OUTPUT_EMB_NPY}")
print(f"- FAISS   : {OUTPUT_FAISS}")

# ------------------------------
# Cell 8
# ------------------------------
# ============================================================
# Vector Search: "R&D사업" (Google Colab)
# ============================================================


import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ------------------------------------------------------------
# 1. 경로 설정
# ------------------------------------------------------------
META_CSV = "/content/policy_news_chunks_meta.csv"
EMB_NPY  = "/content/policy_news_chunks_embeddings.npy"
FAISS_IDX = "/content/policy_news_faiss.index"

MODEL_NAME = "intfloat/multilingual-e5-base"

TOP_K = 5   # 검색 결과 개수

# ------------------------------------------------------------
# 2. 데이터 로드
# ------------------------------------------------------------
chunk_df = pd.read_csv(META_CSV)
embeddings = np.load(EMB_NPY)
index = faiss.read_index(FAISS_IDX)

print(f"[INFO] Chunk 수: {len(chunk_df)}")
print(f"[INFO] Embedding shape: {embeddings.shape}")

# ------------------------------------------------------------
# 3. 임베딩 모델 로드
# ------------------------------------------------------------
model = SentenceTransformer(MODEL_NAME)

# ------------------------------------------------------------
# 4. 검색어 설정
# ------------------------------------------------------------
query = "R&D사업"

# e5 모델 규칙: query prefix 필수
query_vec = model.encode(
    [f"query: {query}"],
    normalize_embeddings=True
)

# ------------------------------------------------------------
# 5. 벡터 검색
# ------------------------------------------------------------
scores, indices = index.search(query_vec, TOP_K)

# ------------------------------------------------------------
# 6. 결과 출력
# ------------------------------------------------------------
print(f"\n[QUERY] {query}")
print("-" * 80)

for rank, idx in enumerate(indices[0]):
    row = chunk_df.iloc[idx]
    score = scores[0][rank]

    print(f"\n[TOP {rank+1}]  score = {score:.4f}")
    print(f"제목      : {row.get('title', '')}")
    print(f"문서 ID  : {row['doc_id']}, Chunk ID: {row['chunk_id']}")
    print("본문 발췌:")
    print(row['text'][:500])
