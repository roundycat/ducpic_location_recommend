import os
import re
import json
import math
import threading
import webbrowser
from typing import Any, Dict, List, Tuple, Optional

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from rich import print
from rich.table import Table

from google import genai
from google.genai import types

from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse

load_dotenv(dotenv_path=".env")

# -----------------------------
# ENV / 기본 설정
# -----------------------------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.0-flash")

CSE_API_KEY = os.environ.get("GOOGLE_CSE_API_KEY", "")
CSE_CX = os.environ.get("GOOGLE_CSE_CX", "")

PLACES_API_KEY = os.environ.get("GOOGLE_PLACES_API_KEY", "")

SEARCH_TOTAL_RESULTS = int(os.environ.get("SEARCH_TOTAL_RESULTS", "100"))   # ✅ 늘림 권장
FETCH_TOP_PAGES = int(os.environ.get("FETCH_TOP_PAGES", "50"))            # ✅ 유효 페이지 목표치 (대폭 증가)
MIN_PAGE_TEXT_CHARS = int(os.environ.get("MIN_PAGE_TEXT_CHARS", "200"))   # ✅ 너무 짧은 페이지 제외 (더 완화)

RADIUS_M = int(os.environ.get("RADIUS_M", "20000"))  # Places bias radius (20km로 증가)
MAX_DISTANCE_KM = float(os.environ.get("MAX_DISTANCE_KM", "10"))  # final hard filter (10km)

STRICT_EVIDENCE_MATCH = os.environ.get("STRICT_EVIDENCE_MATCH", "true").lower() in ("1", "true", "yes", "y")
GPS_PORT = int(os.environ.get("GPS_PORT", "8787"))

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PLACES_TEXTSEARCH_URL = "https://places.googleapis.com/v1/places:searchText"
PLACES_FIELD_MASK = (
    "places.id,places.displayName,places.formattedAddress,places.location,"
    "places.types,places.rating,places.userRatingCount"
)

GEOCODE_REVERSE_URL = "https://maps.googleapis.com/maps/api/geocode/json"

# -----------------------------
# CSE 품질 강제(필터)
# -----------------------------
BAD_DOMAIN_PATTERNS = [
    "huggingface.co",
    "namu.wiki",
    "namu.moe",
    "tiktok.com",
    "daangn.com",
]

BAD_EXTENSIONS = (
    ".txt", ".xml", ".json", ".csv", ".zip", ".pdf", ".mp4", ".m3u8"
)

# 관광/여행과 무관한 키워드(강제 제외)
NEGATIVE_QUERY_KEYWORDS = [
    "학원", "강사", "레슨", "수강", "프로필", "lecturer", "profile", "dance", "lesson"
]


def is_bad_link(url: str) -> bool:
    if not url:
        return True
    u = url.lower()
    if any(d in u for d in BAD_DOMAIN_PATTERNS):
        return True
    if any(ext in u for ext in BAD_EXTENSIONS):
        return True
    return False


# -----------------------------
# 유틸
# -----------------------------
def normalize_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def clean_text(s: str) -> str:
    s = re.sub(r"\s+", " ", s)
    return s.strip()


# -----------------------------
# GPS 좌표 수신 (로컬 HTTP 서버)
# -----------------------------
class GPSState:
    def __init__(self):
        self.lat: Optional[float] = None
        self.lon: Optional[float] = None
        self.err: Optional[str] = None
        self.event = threading.Event()


GPS = GPSState()


def gps_capture_page_html(port: int) -> str:
    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8"/>
  <title>GPS 위치 확인</title>
  <style>
    body {{ font-family: system-ui, -apple-system, sans-serif; padding: 24px; }}
    .box {{ max-width: 720px; margin: 0 auto; border: 1px solid #ddd; border-radius: 14px; padding: 18px; }}
    .muted {{ color: #666; }}
    button {{ padding: 10px 14px; border-radius: 10px; border: 1px solid #ddd; background: #fff; cursor: pointer; }}
    pre {{ background: #f7f7f7; padding: 12px; border-radius: 12px; overflow: auto; }}
  </style>
</head>
<body>
  <div class="box">
    <h2>브라우저 GPS 위치 권한 요청</h2>
    <p class="muted">
      허용을 누르면 현재 GPS 좌표를 파이썬으로 전달합니다.
    </p>

    <button onclick="send()">현재 위치 보내기</button>
    <p id="status" class="muted"></p>
    <pre id="out"></pre>
  </div>

<script>
async function send() {{
  const status = document.getElementById("status");
  const out = document.getElementById("out");
  out.textContent = "";
  status.textContent = "GPS 위치 요청 중...";

  if (!navigator.geolocation) {{
    status.textContent = "이 브라우저는 geolocation을 지원하지 않습니다.";
    return;
  }}

  navigator.geolocation.getCurrentPosition(async (pos) => {{
    const payload = {{
      lat: pos.coords.latitude,
      lon: pos.coords.longitude,
      accuracy_m: pos.coords.accuracy
    }};
    status.textContent = "좌표 수신됨. 서버로 전송 중...";
    out.textContent = JSON.stringify(payload, null, 2);

    const res = await fetch("/submit", {{
      method: "POST",
      headers: {{ "Content-Type": "application/json" }},
      body: JSON.stringify(payload)
    }});

    if (res.ok) {{
      status.textContent = "전송 완료! 이제 이 탭을 닫고 터미널로 돌아가도 됩니다.";
    }} else {{
      status.textContent = "전송 실패: " + res.status;
    }}
  }}, (err) => {{
    status.textContent = "GPS 권한/획득 실패: " + err.message;
  }}, {{
    enableHighAccuracy: true,
    timeout: 15000,
    maximumAge: 0
  }});
}}
</script>
</body>
</html>
"""


class GPSHandler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: str, content_type: str = "text/html; charset=utf-8"):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body.encode("utf-8"))

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/" or path == "/index.html":
            self._send(200, gps_capture_page_html(GPS_PORT))
        else:
            self._send(404, "not found", "text/plain; charset=utf-8")

    def do_POST(self):
        path = urlparse(self.path).path
        if path != "/submit":
            self._send(404, "not found", "text/plain; charset=utf-8")
            return

        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8")
        try:
            data = json.loads(raw)
            GPS.lat = float(data.get("lat"))
            GPS.lon = float(data.get("lon"))
            GPS.err = None
            GPS.event.set()
            self._send(200, "ok", "text/plain; charset=utf-8")
        except Exception as e:
            GPS.err = str(e)
            GPS.event.set()
            self._send(400, "bad request", "text/plain; charset=utf-8")

    def log_message(self, format, *args):
        return


def get_location_by_browser_gps(port: int) -> Tuple[float, float, Dict[str, Any]]:
    server = HTTPServer(("127.0.0.1", port), GPSHandler)

    def run_server():
        server.serve_forever()

    th = threading.Thread(target=run_server, daemon=True)
    th.start()

    url = f"http://127.0.0.1:{port}/"
    print(f"[cyan]브라우저에서 GPS 권한을 허용해 주세요[/cyan]: {url}")
    try:
        webbrowser.open(url)
    except Exception:
        pass

    GPS.event.wait(timeout=120)
    server.shutdown()

    if GPS.err:
        raise RuntimeError(f"GPS 수신 실패: {GPS.err}")
    if GPS.lat is None or GPS.lon is None:
        raise RuntimeError("GPS 수신 타임아웃(120초). 브라우저에서 '현재 위치 보내기'를 눌러주세요.")

    return float(GPS.lat), float(GPS.lon), {"provider": "browser_gps"}


# -----------------------------
# Reverse Geocoding (구/시까지만 힌트)
# -----------------------------
def reverse_geocode_admin(lat: float, lon: float) -> Dict[str, str]:
    params = {
        "latlng": f"{lat},{lon}",
        "language": "ko",
        "key": PLACES_API_KEY,
        "result_type": "administrative_area_level_2|locality|administrative_area_level_1",
    }
    r = requests.get(GEOCODE_REVERSE_URL, params=params, timeout=15)
    if r.status_code >= 400:
        return {}
    data = r.json()
    results = data.get("results", []) or []
    if not results:
        return {}
    best = results[0]
    comps = best.get("address_components", []) or []
    out = {"formatted": best.get("formatted_address", "")}

    def pick(type_candidates: List[str]) -> str:
        for c in comps:
            c_types = c.get("types", [])
            if any(t in c_types for t in type_candidates):
                return c.get("long_name") or ""
        return ""

    out["district"] = pick(["administrative_area_level_2"])
    out["city"] = pick(["locality"])
    out["province"] = pick(["administrative_area_level_1"])
    out["country"] = pick(["country"])
    return out


def make_location_hint_big(admin: Dict[str, str]) -> str:
    # ✅ 도로명/세부 제거: 광역 단위만
    parts = []
    for k in ["province", "city", "district"]:
        v = (admin.get(k) or "").strip()
        if v and v not in parts:
            parts.append(v)
    return " ".join(parts).strip()


# -----------------------------
# 연예인 파싱/직접 관련 필터
# -----------------------------
def guess_celebrity_from_query(user_query: str) -> str:
    uq = normalize_space(user_query)
    if not uq:
        return ""
    # 첫 토큰을 연예인명으로
    return uq.split()[0].strip()


def is_directly_related(celebrity: str, evidence_text: str) -> bool:
    celebrity = (celebrity or "").strip()
    if not celebrity:
        return False
    ev = evidence_text or ""
    if STRICT_EVIDENCE_MATCH:
        # STRICT 모드: 연예인 이름이 포함되어야 함
        if celebrity not in ev:
            return False
    else:
        # 비STRICT 모드: evidence_text가 비어있지 않으면 OK
        if not ev or len(ev.strip()) < 10:
            return False
    return True


# -----------------------------
# CSE Query 생성(관광지 중심 + 제외 키워드)
# -----------------------------
def build_cse_query_variants(user_query: str, celebrity: str, loc_big: str) -> List[str]:
    """여러 쿼리 변형을 생성하여 더 많은 결과 확보 - 더 단순하고 넓게"""
    celeb = f'"{celebrity}"'
    variants = []
    
    # 도메인 제외 (필수만)
    neg_sites = " ".join([f"-site:{d}" for d in BAD_DOMAIN_PATTERNS])
    # 키워드 제외 (필수만)
    neg_kw = " ".join([f"-{k}" for k in NEGATIVE_QUERY_KEYWORDS])
    
    # 1. 가장 단순한 쿼리 (연예인명 + 위치만)
    if loc_big:
        simple = f'{celeb} {loc_big}'
        variants.append(normalize_space(f"{simple} {neg_sites} {neg_kw}"))
    
    # 2. 연예인명만 (가장 넓은 검색)
    variants.append(normalize_space(f"{celeb} {neg_sites} {neg_kw}"))
    
    # 3. 기본 패턴들 (더 단순하게)
    base_patterns = [
        f'{celeb} 촬영지',
        f'{celeb} 방문',
        f'{celeb} 다녀온',
        f'{celeb} 갔다',
        f'{celeb} 브이로그',
        f'{celeb} 카페',
        f'{celeb} 맛집',
        f'{celeb} 관광지',
        f'{celeb} 명소',
    ]
    
    for pattern in base_patterns:
        core = pattern
        if loc_big:
            core += f" {loc_big}"
        variants.append(normalize_space(f"{core} {neg_sites} {neg_kw}"))
    
    # 4. 사용자 쿼리 포함 (원본 쿼리도 시도)
    if user_query and user_query != celebrity:
        user_q = normalize_space(user_query)
        if loc_big:
            variants.append(normalize_space(f"{user_q} {loc_big} {neg_sites} {neg_kw}"))
        variants.append(normalize_space(f"{user_q} {neg_sites} {neg_kw}"))
    
    # 중복 제거
    return list(dict.fromkeys(variants))


def build_cse_query(user_query: str, celebrity: str, loc_big: str) -> str:
    """기존 호환성을 위한 함수"""
    variants = build_cse_query_variants(user_query, celebrity, loc_big)
    return variants[0] if variants else ""


# -----------------------------
# 1) CSE 검색(페이지네이션 + 필터)
# -----------------------------
def cse_search(query: str, total: int = 80) -> List[Dict[str, Any]]:
    if not CSE_API_KEY or not CSE_CX:
        raise RuntimeError("GOOGLE_CSE_API_KEY, GOOGLE_CSE_CX 환경변수가 필요합니다.")

    total = max(1, min(total, 100))
    results: List[Dict[str, Any]] = []
    start = 1
    seen = set()

    while len(results) < total:
        num = min(10, total - len(results))
        params = {"key": CSE_API_KEY, "cx": CSE_CX, "q": query, "num": num, "start": start, "hl": "ko"}
        r = requests.get("https://www.googleapis.com/customsearch/v1", params=params, timeout=15)
        if r.status_code >= 400:
            raise RuntimeError(f"CSE error {r.status_code}: {r.text}")

        data = r.json()
        items = data.get("items", []) or []
        if not items:
            break

        for it in items:
            link = it.get("link") or ""
            if not link or link in seen:
                continue
            if is_bad_link(link):
                continue
            seen.add(link)
            results.append({"title": it.get("title"), "link": link, "snippet": it.get("snippet")})

        start += len(items)
        if len(items) < num:
            break

    return results


# -----------------------------
# 2) 웹 페이지 텍스트 수집
# -----------------------------
def fetch_page_text(url: str, max_chars: int = 8000) -> str:
    try:
        r = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"})
        if r.status_code >= 400:
            return ""
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script", "style", "noscript", "iframe", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator=" ")
        text = clean_text(text)
        return text[:max_chars]
    except Exception:
        return ""


# -----------------------------
# 3) Gemini로 장소 추출(근거 기반)
# -----------------------------
def gemini_extract_places_from_sources(
    client: genai.Client,
    user_query: str,
    celebrity: str,
    location_hint: str,
    sources: List[Dict[str, Any]],
) -> Dict[str, Any]:
    schema = {
        "type": "object",
        "properties": {
            "places": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "place_name": {"type": "string"},
                        "area_hint": {"type": "string"},
                        "evidence_url": {"type": "string"},
                        "evidence_text": {"type": "string"},
                        "category_hint": {"type": "string"},
                    },
                    "required": ["place_name", "evidence_url", "evidence_text"],
                },
            }
        },
        "required": ["places"],
    }

    instruction = {
        "role": "너는 웹 페이지에서 장소를 추출하는 정보추출기다. 매우 적극적으로 추출해야 한다.",
        "task": f"아래 sources에서 '{celebrity}'와 관련된 모든 장소를 가능한 한 많이 추출하라.",
        "rules": [
            "근거가 sources에 실제로 존재할 때만 추출(추측 금지).",
            "evidence_text는 연예인과 장소가 연결된 문장을 사용(연예인 이름이 없어도 문맥상 연결되면 OK).",
            "place_name은 장소명만 간결히 추출 (불완전해도 OK).",
            "area_hint는 가능하면 'OO시/OO구'까지만 추출 (없어도 OK, 추측해도 OK).",
            "모든 종류의 장소를 추출 (카페/맛집/식당/레스토랑/관광지/명소/공원/전시/박물관/미술관/갤러리/전시관/문화센터/체험관/쇼핑몰/마트/영화관/극장/공연장/체육관/수영장/스키장/리조트/호텔/펜션/게스트하우스 등).",
            "학원/레슨/프로필/강사 같은 성격이면 제외.",
            "반드시 최소 30개 이상 추출해야 함 (가능하면 50개 이상).",
            "장소명이 불명확해도 추출 시도 (나중에 지오코딩으로 보정 가능).",
            "같은 장소가 여러 번 언급되면 모두 추출해도 OK (중복 제거는 나중에).",
            "장소명이 부분적으로만 나와도 추출 (예: '스타벅스'만 나와도 OK).",
            "위치 정보가 없어도 장소명만 추출해도 OK."
        ],
        "celebrity": celebrity,
        "user_query": user_query,
        "user_location_hint": location_hint,
        "sources": sources,
        "important": "반드시 최소 30개 이상의 장소를 추출해야 합니다. 가능한 한 많이 추출하세요!"
    }

    resp = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[types.Content(role="user", parts=[types.Part(text=json.dumps(instruction, ensure_ascii=False))])],
        config={"response_mime_type": "application/json", "response_json_schema": schema},
    )
    return json.loads(resp.text)


# -----------------------------
# 4) Places Text Search로 지오코딩
# -----------------------------
def places_text_search(place_name: str, area_hint: str, bias_lat: float, bias_lon: float, radius_m: int) -> Optional[Dict[str, Any]]:
    # 여러 시도 전략 (더 많은 변형)
    queries = []
    
    # 1. area_hint 포함
    if area_hint:
        queries.append(f"{place_name} {area_hint}")
        # area_hint의 일부만 사용
        area_parts = area_hint.split()
        if len(area_parts) > 1:
            queries.append(f"{place_name} {area_parts[0]}")
    
    # 2. 원본 장소명
    queries.append(place_name)
    
    # 3. 장소명 정리 (괄호, 특수문자 제거)
    clean_name = re.sub(r'[\(\)\[\]【】]', '', place_name).strip()
    if clean_name and clean_name != place_name:
        queries.append(clean_name)
        if area_hint:
            queries.append(f"{clean_name} {area_hint}")
    
    # 4. 장소명에서 불필요한 단어 제거
    remove_words = ["카페", "식당", "맛집", "레스토랑", "공원", "박물관", "미술관"]
    for word in remove_words:
        if word in place_name:
            simplified = place_name.replace(word, "").strip()
            if simplified and simplified != place_name:
                queries.append(simplified)
                if area_hint:
                    queries.append(f"{simplified} {area_hint}")
            break
    
    # 중복 제거
    queries = list(dict.fromkeys(queries))  # 순서 유지하면서 중복 제거

    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": PLACES_API_KEY,
        "X-Goog-FieldMask": PLACES_FIELD_MASK,
    }

    for q in queries:
        # radius를 점진적으로 늘려가며 시도
        for attempt_radius in [radius_m, radius_m * 2, radius_m * 3]:
            body = {
                "textQuery": q,
                "maxResultCount": 1,
                "locationBias": {"circle": {"center": {"latitude": bias_lat, "longitude": bias_lon}, "radius": attempt_radius}},
                "languageCode": "ko",
            }

            try:
                r = requests.post(PLACES_TEXTSEARCH_URL, headers=headers, json=body, timeout=20)
                if r.status_code >= 400:
                    continue
                
                data = r.json()
                places = data.get("places", []) or []
                if not places:
                    continue

                p = places[0]
                loc = p.get("location", {})
                lat = loc.get("latitude")
                lon = loc.get("longitude")
                name_obj = p.get("displayName", {})
                name = name_obj.get("text") if isinstance(name_obj, dict) else str(name_obj)

                if lat is None or lon is None:
                    continue

                return {
                    "place_id": p.get("id"),
                    "name": name,
                    "address": p.get("formattedAddress"),
                    "lat": float(lat),
                    "lon": float(lon),
                    "types": p.get("types", []) or [],
                    "rating": p.get("rating"),
                    "user_ratings_total": p.get("userRatingCount"),
                }
            except Exception:
                continue

    return None


# -----------------------------
# 출력(표/지도)
# -----------------------------
def show_table(rows: List[Dict[str, Any]], title: str):
    t = Table(title=title)
    t.add_column("#", justify="right")
    t.add_column("장소", overflow="fold")
    t.add_column("주소", overflow="fold")
    t.add_column("거리(km)", justify="right")
    t.add_column("근거(요약)", overflow="fold")
    t.add_column("출처", overflow="fold")

    for i, r in enumerate(rows, 1):
        dist = r.get("distance_km")
        dist_str = f"{dist:.2f}" if isinstance(dist, (int, float)) else ""
        t.add_row(
            str(i),
            r.get("name") or "",
            r.get("address", "") or "",
            dist_str,
            (r.get("evidence_text") or "")[:90],
            (r.get("evidence_url") or "")[:65],
        )
    print(t)


def write_map_html(places: List[Dict[str, Any]], center_lat: float, center_lon: float):
    html = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8"/>
  <title>연예인 관련 장소 지도 (GPS 10km)</title>
  <style>
    html, body {{ height: 100%; margin: 0; }}
    #map {{ height: 100%; width: 100%; }}
    .panel {{
      position: absolute; top: 12px; left: 12px; z-index: 5;
      background: white; padding: 10px 12px; border-radius: 10px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.18);
      max-width: 520px; font-family: system-ui, sans-serif; font-size: 14px;
    }}
    .panel small {{ color: #666; }}
  </style>
</head>
<body>
  <div class="panel">
    <div><b>연예인 관련 장소(근거 기반) - GPS 기준 10km 이내만</b></div>
    <div><small>내 위치는 브라우저 권한이 허용되면 표시됩니다.</small></div>
  </div>
  <div id="map"></div>

  <script>
    const places = {json.dumps(places, ensure_ascii=False)};

    function initMap() {{
      const center = {{ lat: {center_lat}, lng: {center_lon} }};
      const map = new google.maps.Map(document.getElementById("map"), {{
        zoom: 13,
        center
      }});

      if (navigator.geolocation) {{
        navigator.geolocation.getCurrentPosition((pos) => {{
          const me = {{ lat: pos.coords.latitude, lng: pos.coords.longitude }};
          new google.maps.Marker({{
            position: me,
            map,
            title: "내 현재 위치"
          }});
        }});
      }}

      places.forEach((p) => {{
        const m = new google.maps.Marker({{
          position: {{ lat: p.lat, lng: p.lon }},
          map,
          title: p.name
        }});

        const content = `
          <div style="max-width:420px;">
            <div style="font-weight:700;">${{p.name}}</div>
            <div>${{p.address || ""}}</div>
            <div style="margin-top:6px;"><b>거리</b>: ${{(p.distance_km ?? "").toString()}} km</div>
            <div style="margin-top:6px;"><b>근거</b>: ${{p.evidence_text || ""}}</div>
            <div style="margin-top:6px;">
              <a href="${{p.evidence_url}}" target="_blank">출처 열기</a>
            </div>
          </div>
        `;
        const iw = new google.maps.InfoWindow({{ content }});
        m.addListener("click", () => iw.open(map, m));
      }});
    }}
  </script>

  <script async defer
    src="https://maps.googleapis.com/maps/api/js?key={PLACES_API_KEY}&callback=initMap">
  </script>
</body>
</html>
"""
    out_path = os.path.join(OUTPUT_DIR, "map.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[green]지도 HTML 생성 완료[/green]: {out_path}")


# -----------------------------
# main
# -----------------------------
def main():
    if not GEMINI_API_KEY:
        raise SystemExit("GEMINI_API_KEY가 필요합니다.")
    if not CSE_API_KEY or not CSE_CX:
        raise SystemExit("GOOGLE_CSE_API_KEY, GOOGLE_CSE_CX가 필요합니다.")
    if not PLACES_API_KEY:
        raise SystemExit("GOOGLE_PLACES_API_KEY가 필요합니다.")

    user_query = normalize_space(input("검색 주제 입력 (예: '오마이걸 방문 여행지'): "))
    if not user_query:
        user_query = "오마이걸 방문 여행지"

    celebrity = guess_celebrity_from_query(user_query)
    print(f"[cyan]직접 관련 필터 기준 연예인[/cyan]: '{celebrity}' (STRICT={STRICT_EVIDENCE_MATCH})")
    print(f"[cyan]GPS 기반 10km 강제[/cyan]: 브라우저에서 좌표를 받아 사용합니다.")

    # GPS
    bias_lat, bias_lon, meta = get_location_by_browser_gps(GPS_PORT)
    print("[bold]내 위치(GPS)[/bold]", bias_lat, bias_lon, meta)

    # Reverse geocode -> 광역 힌트만
    admin = reverse_geocode_admin(bias_lat, bias_lon)
    loc_big = make_location_hint_big(admin)
    print(f"[cyan]검색 위치 힌트(광역)[/cyan]: {loc_big}")
    print(f"[cyan]거리 강제[/cyan]: Places bias {RADIUS_M}m + 최종 {MAX_DISTANCE_KM}km 컷")

    # CSE query (여러 변형 시도)
    query_variants = build_cse_query_variants(user_query=user_query, celebrity=celebrity, loc_big=loc_big)
    print(f"[cyan]CSE 쿼리 변형 {len(query_variants)}개 생성[/cyan]")
    
    # 여러 쿼리로 검색하여 결과 병합
    all_results = []
    seen_urls = set()
    
    for i, query in enumerate(query_variants, 1):
        print(f"[dim]쿼리 {i}/{len(query_variants)} 실행[/dim]: {query[:80]}...")
        try:
            # 각 쿼리당 더 많이 수집 (최대 100개까지)
            query_results = cse_search(query, total=min(SEARCH_TOTAL_RESULTS, 100))
            added = 0
            for r in query_results:
                url = r.get("link", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    all_results.append(r)
                    added += 1
            print(f"[dim]쿼리 {i} 결과: {len(query_results)}개 (신규: {added}개, 누적: {len(all_results)}개)[/dim]")
        except Exception as e:
            print(f"[yellow]쿼리 {i} 실패: {e}[/yellow]")
            continue
    
    results = all_results[:SEARCH_TOTAL_RESULTS]  # 최종적으로 제한
    print(f"[cyan]검색 결과 누적 {len(results)}개[/cyan]")
    for r in results[:10]:
        print("-", r["title"], r["link"])
    if len(results) > 10:
        print(f"[dim]... (총 {len(results)}개)[/dim]")

    # 페이지 수집: "유효 텍스트" 기준으로 목표치 채움 (더 많이 수집)
    sources = []
    need = min(FETCH_TOP_PAGES, len(results))
    skipped_bad = 0
    skipped_short = 0
    failed_fetch = 0
    
    for r in results:
        if len(sources) >= need:
            break
        url = r.get("link", "")
        if is_bad_link(url):
            skipped_bad += 1
            continue
        
        page_text = fetch_page_text(url, max_chars=8000)  # 더 많은 텍스트 수집
        if not page_text:
            failed_fetch += 1
            continue
        if len(page_text) < MIN_PAGE_TEXT_CHARS:
            skipped_short += 1
            if len(sources) < need * 0.8:  # 목표의 80% 미만이면 짧은 것도 수집
                print(f"[dim]페이지 수집 (짧지만 포함: {len(page_text)} < {MIN_PAGE_TEXT_CHARS})[/dim]: {url}")
            else:
                print(f"[dim]페이지 스킵 (짧음: {len(page_text)} < {MIN_PAGE_TEXT_CHARS})[/dim]: {url}")
                continue
        
        sources.append({
            "title": r.get("title"),
            "link": url,
            "snippet": r.get("snippet"),
            "page_text": page_text,
        })
        print(f"[dim]페이지 수집 OK {len(sources)}/{need}[/dim]: {url} (chars={len(page_text)})")
    
    if skipped_bad > 0 or skipped_short > 0 or failed_fetch > 0:
        print(f"[yellow]페이지 수집 통계: OK={len(sources)}, 스킵(나쁜링크)={skipped_bad}, 스킵(짧음)={skipped_short}, 실패={failed_fetch}[/yellow]")

    if not sources:
        raise SystemExit("유효한 페이지 텍스트를 수집하지 못했습니다. SEARCH_TOTAL_RESULTS/FETCH_TOP_PAGES를 늘리거나 CSE 설정(CX)을 확인하세요.")

    # Gemini 추출 (여러 배치로 나눠서 더 많이 추출)
    client = genai.Client(api_key=GEMINI_API_KEY)
    all_places_raw = []
    
    # sources를 여러 배치로 나눠서 추출 (각 배치당 더 집중적으로)
    batch_size = max(10, len(sources) // 3)  # 최소 10개씩, 최대 3개 배치
    for batch_idx in range(0, len(sources), batch_size):
        batch_sources = sources[batch_idx:batch_idx + batch_size]
        print(f"[cyan]Gemini 추출 배치 {batch_idx // batch_size + 1}/{(len(sources) + batch_size - 1) // batch_size} (sources: {len(batch_sources)}개)[/cyan]")
        try:
            extracted = gemini_extract_places_from_sources(client, user_query, celebrity, loc_big, batch_sources)
            batch_places = extracted.get("places", []) or []
            all_places_raw.extend(batch_places)
            print(f"[dim]배치 추출 결과: {len(batch_places)}개 (누적: {len(all_places_raw)}개)[/dim]")
        except Exception as e:
            print(f"[yellow]배치 추출 실패: {e}[/yellow]")
            continue
    
    places_raw = all_places_raw
    out1 = os.path.join(OUTPUT_DIR, "places.json")
    with open(out1, "w", encoding="utf-8") as f:
        json.dump({"places": places_raw, "sources_count": len(sources)}, f, ensure_ascii=False, indent=2)
    print(f"[green]장소 후보 추출 저장[/green]: {out1} (count={len(places_raw)})")

    if not places_raw:
        raise SystemExit("근거 기반으로 추출된 장소가 없습니다. (검색 결과 품질이 낮거나 주변 10km에 실제로 없을 수 있음)")

    # 직접 관련 필터(완화된 버전)
    if STRICT_EVIDENCE_MATCH:
        places_raw_filtered = [p for p in places_raw if is_directly_related(celebrity, p.get("evidence_text", ""))]
        print(f"[cyan]직접 관련 필터 (STRICT={STRICT_EVIDENCE_MATCH})[/cyan]: {len(places_raw)}개 -> {len(places_raw_filtered)}개")
        if len(places_raw_filtered) < len(places_raw) * 0.3:  # 30% 미만이면 필터 너무 강함
            print(f"[yellow]필터 결과가 너무 적습니다. 필터를 완화합니다.[/yellow]")
            places_raw = places_raw  # 필터 적용 안 함
        else:
            places_raw = places_raw_filtered
    else:
        # 비STRICT 모드: evidence_text가 있으면 OK
        places_raw = [p for p in places_raw if p.get("evidence_text", "").strip()]
        print(f"[cyan]직접 관련 필터 (STRICT=false)[/cyan]: {len(places_raw)}개")
    
    if not places_raw:
        print(f"[yellow]경고: 필터 후 장소가 없습니다. 필터를 완전히 제거합니다.[/yellow]")
        # 필터 제거하고 원본 사용
        places_raw = all_places_raw if 'all_places_raw' in locals() else places_raw

    # 지오코딩 (중복 제거 포함, 실패해도 일단 저장)
    geocoded = []
    geocoded_place_ids = set()  # 중복 제거용
    geocode_failed = []
    geocode_failed_with_info = []  # 실패했지만 정보는 있는 것들
    
    print(f"[cyan]지오코딩 시작: {len(places_raw)}개 장소[/cyan]")
    for i, p in enumerate(places_raw, 1):
        place_name = normalize_space(p.get("place_name", ""))
        area_hint = normalize_space(p.get("area_hint", ""))
        if not place_name:
            geocode_failed.append({"place": place_name, "reason": "빈 장소명"})
            continue

        g = places_text_search(place_name, area_hint, bias_lat, bias_lon, RADIUS_M)
        if not g:
            # 지오코딩 실패해도 일단 정보는 저장 (나중에 거리 필터에서 제외)
            geocode_failed_with_info.append({
                "place_name": place_name,
                "area_hint": area_hint,
                "evidence_url": p.get("evidence_url"),
                "evidence_text": p.get("evidence_text"),
                "reason": "Places API 검색 실패"
            })
            if i <= 10 or len(geocode_failed) <= 5:
                print(f"[dim]지오코딩 실패 {i}/{len(places_raw)}[/dim]: {place_name} ({area_hint})")
            continue

        # 중복 제거 (place_id 기준)
        place_id = g.get("place_id")
        if place_id and place_id in geocoded_place_ids:
            continue
        if place_id:
            geocoded_place_ids.add(place_id)

        g["evidence_url"] = p.get("evidence_url")
        g["evidence_text"] = p.get("evidence_text")
        g["area_hint"] = area_hint
        geocoded.append(g)
        if i <= 10 or len(geocoded) % 5 == 0:
            print(f"[dim]지오코딩 성공 {i}/{len(places_raw)} (누적: {len(geocoded)})[/dim]: {place_name} -> {g.get('name')} ({g.get('address', '')[:50]})")
    
    print(f"[cyan]지오코딩 완료: 성공 {len(geocoded)}개, 실패 {len(geocode_failed_with_info)}개[/cyan]")
    
    # 지오코딩 실패한 것들도 저장 (참고용)
    if geocode_failed_with_info:
        print(f"[yellow]지오코딩 실패한 장소 {len(geocode_failed_with_info)}개 (참고용으로 저장)[/yellow]")

    # 최종 10km hard filter (거리 계산 및 필터링)
    final_rows = []
    outside_radius = []
    
    for g in geocoded:
        d = haversine_km(bias_lat, bias_lon, g["lat"], g["lon"])
        g["distance_km"] = round(d, 3)
        if d <= MAX_DISTANCE_KM:
            final_rows.append(g)
        else:
            outside_radius.append(g)

    final_rows = sorted(final_rows, key=lambda x: x["distance_km"])
    print(f"[cyan]최종 거리 필터[/cyan]: {MAX_DISTANCE_KM}km 이내 -> {len(final_rows)}개, 범위 밖 -> {len(outside_radius)}개")
    
    # 범위 밖이지만 가까운 것들도 표시 (참고용)
    if len(final_rows) < 5 and outside_radius:
        outside_radius_sorted = sorted(outside_radius, key=lambda x: x["distance_km"])
        print(f"[yellow]참고: {MAX_DISTANCE_KM}km 밖이지만 가까운 장소 {min(5, len(outside_radius_sorted))}개[/yellow]")
        for g in outside_radius_sorted[:5]:
            print(f"  - {g.get('name')}: {g.get('distance_km')}km")

    out2 = os.path.join(OUTPUT_DIR, "places_geocoded.json")
    with open(out2, "w", encoding="utf-8") as f:
        json.dump(
            {"query": user_query, "celebrity": celebrity, "loc_big": loc_big,
             "gps_location": {"lat": bias_lat, "lon": bias_lon}, "places": final_rows},
            f, ensure_ascii=False, indent=2
        )
    print(f"[green]좌표 확정 결과 저장[/green]: {out2}")

    if not final_rows:
        raise SystemExit("최종 10km 조건을 만족하는 장소가 없습니다. (주변 10km에 실제로 없거나 근거가 부족)")

    show_table(final_rows, "연예인 관련 장소 + 좌표 확정 (GPS 10km 이내)")
    write_map_html(final_rows, bias_lat, bias_lon)

    print("\n[bold]다음 파일을 확인하세요[/bold]")
    print("-", out1)
    print("-", out2)
    print("-", os.path.join(OUTPUT_DIR, "map.html"))
    print("\n지도 안 뜨면 아래로 로컬 서버 실행 후 접속:")
    print("  python -m http.server 8000")
    print("  http://localhost:8000/output/map.html")


if __name__ == "__main__":
    main()