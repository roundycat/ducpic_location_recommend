import os
import json
import math
import socket
import sys
import re
from typing import Any, Dict, List, Optional, Tuple
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import requests
from google import genai
from google.genai import types
from dotenv import load_dotenv

# pipeline_functions 모듈 import
try:
    from pipeline_functions import (
        normalize_space, guess_celebrity_from_query, build_cse_query_variants,
        cse_search, naver_search, naver_blog_search, multi_search,
        fetch_page_text, gemini_extract_places_from_sources,
        places_text_search, reverse_geocode_admin, make_location_hint_big,
        is_directly_related, is_actually_visited, haversine_km as haversine_km_pipeline,
        RADIUS_M, MAX_DISTANCE_KM, SEARCH_TOTAL_RESULTS, FETCH_TOP_PAGES,
        MIN_PAGE_TEXT_CHARS, CSE_API_KEY, CSE_CX, STRICT_EVIDENCE_MATCH
    )
    PIPELINE_AVAILABLE = True
except ImportError:
    PIPELINE_AVAILABLE = False

# query_intent_analyzer 모듈 import
try:
    from query_intent_analyzer import analyze_query_intent, filter_places_by_intent, should_include_place
    INTENT_ANALYZER_AVAILABLE = True
except ImportError:
    INTENT_ANALYZER_AVAILABLE = False
    # Fallback 함수들
    def analyze_query_intent(user_query: str):
        class DummyIntent:
            category = "general"
            required_types = set()
            excluded_types = set()
            excluded_keywords = set()
            priority_types = set()
            max_distance_km = 20.0
            strict_filtering = False
        return DummyIntent()
    
    def should_include_place(place, intent):
        return (True, 1.0)
    
    def filter_places_by_intent(places, intent, max_distance_km):
        return places


def normalize_place_name(name: str) -> str:
    """장소명 정규화 (중복 제거용)"""
    if not name:
        return ""
    # 소문자 변환, 공백 제거, 특수문자 제거
    normalized = re.sub(r'[^\w가-힣]', '', name.lower().strip())
    # 지점명 제거
    normalized = re.sub(r'(점|지점|본점|직영점|체인점)$', '', normalized)
    return normalized

load_dotenv()

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)

# ----------------------------
# 설정
# ----------------------------
PLACES_NEARBY_URL = "https://places.googleapis.com/v1/places:searchNearby"
FIELD_MASK = "places.id,places.displayName,places.formattedAddress,places.location,places.types,places.rating,places.userRatingCount"

DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
DEFAULT_RADIUS_M = int(os.environ.get("DEFAULT_RADIUS_M", "2000"))
DEFAULT_K = int(os.environ.get("DEFAULT_K", "5"))
DEFAULT_MAX_CANDIDATES = int(os.environ.get("DEFAULT_MAX_CANDIDATES", "20"))  # Places API 최대값: 20

# run_pipeline.py 관련 설정은 pipeline_functions 모듈에서 import


# ----------------------------
# 유틸: 거리 계산
# ----------------------------
def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def validate_or_fix_latlon(lat: float, lon: float) -> Tuple[float, float]:
    if -90 <= lat <= 90 and -180 <= lon <= 180:
        return lat, lon
    # 흔한 실수: swap
    if -90 <= lon <= 90 and -180 <= lat <= 180:
        return lon, lat
    raise ValueError(f"좌표 범위 오류: lat={lat}, lon={lon} (lat: [-90,90], lon: [-180,180])")


# ----------------------------
# IP 기반 자동 위치 (옵션)
# ----------------------------
def get_location_by_ip() -> Tuple[float, float, Dict[str, Any]]:
    # 1) ipinfo
    try:
        r = requests.get("https://ipinfo.io/json", timeout=8)
        r.raise_for_status()
        data = r.json()
        loc = data.get("loc")
        if not loc:
            raise RuntimeError(f"ipinfo has no loc: {data}")
        lat_str, lon_str = loc.split(",")
        meta = {
            "provider": "ipinfo.io",
            "ip": data.get("ip"),
            "city": data.get("city"),
            "region": data.get("region"),
            "country": data.get("country"),
            "org": data.get("org"),
        }
        return float(lat_str), float(lon_str), meta
    except Exception as e1:
        # 2) ip-api fallback
        r = requests.get("http://ip-api.com/json", timeout=8)
        r.raise_for_status()
        data = r.json()
        if data.get("status") != "success":
            raise RuntimeError(f"Auto location failed: ipinfo={e1}, ip-api={data}")
        meta = {
            "provider": "ip-api.com",
            "query_ip": data.get("query"),
            "city": data.get("city"),
            "regionName": data.get("regionName"),
            "country": data.get("country"),
            "isp": data.get("isp"),
            "fallback_reason": str(e1),
        }
        return float(data["lat"]), float(data["lon"]), meta


# ----------------------------
# Places API: Nearby Search (New) - 타입 기반만
# ----------------------------
def places_nearby_search(
    lat: float,
    lon: float,
    radius_m: int,
    included_types: Optional[List[str]] = None,
    max_results: int = 20,
) -> Dict[str, Any]:
    # Places API는 최대 20개까지만 허용
    max_results = min(max(1, max_results), 20)
    api_key = os.environ.get("GOOGLE_MAPS_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_MAPS_API_KEY 환경변수가 필요합니다.")

    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": FIELD_MASK,
    }

    body: Dict[str, Any] = {
        "locationRestriction": {
            "circle": {
                "center": {"latitude": lat, "longitude": lon},
                "radius": radius_m,
            }
        },
        "maxResultCount": max_results,
    }

    if included_types:
        body["includedTypes"] = included_types

    r = requests.post(PLACES_NEARBY_URL, headers=headers, json=body, timeout=20)
    if r.status_code >= 400:
        raise RuntimeError(f"Places API error {r.status_code}: {r.text}")
    return r.json()


# ----------------------------
# Gemini: 프롬프트 -> 검색 타입 추출
# ----------------------------
def gemini_extract_types(client: genai.Client, user_prompt: str) -> List[str]:
    schema = {
        "type": "object",
        "properties": {
            "included_types": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["included_types"],
    }

    instruction = {
        "task": "사용자 프롬프트에 맞춰 Nearby Search에 사용할 included_types를 고르라.",
        "rules": [
            "included_types는 1~6개",
            "예시 타입: tourist_attraction, cafe, restaurant, park, museum, art_gallery, bookstore, shopping_mall",
            "프롬프트 조건을 반영하되 과도하게 희귀한 타입은 피하라",
        ],
        "user_prompt": user_prompt
    }

    resp = client.models.generate_content(
        model=DEFAULT_MODEL,
        contents=[types.Content(role="user", parts=[types.Part(text=json.dumps(instruction, ensure_ascii=False))])],
        config={"response_mime_type": "application/json", "response_json_schema": schema},
    )
    data = json.loads(resp.text)
    return data.get("included_types", []) or []


def extract_celebrity_name(prompt: str) -> Optional[str]:
    """프롬프트에서 연예인 이름 추출 (간단한 휴리스틱)"""
    # 한국 아이돌 그룹/연예인 키워드
    kpop_groups = [
        "세븐틴", "방탄소년단", "BTS", "블랙핑크", "BLACKPINK", "뉴진스", "NewJeans",
        "아이브", "IVE", "르세라핌", "LE SSERAFIM", "에스파", "aespa", "트와이스", "TWICE",
        "레드벨벳", "Red Velvet", "오마이걸", "OH MY GIRL", "아이들", "(G)I-DLE",
        "있지", "ITZY", "엔시티", "NCT", "엑소", "EXO", "슈퍼주니어", "Super Junior"
    ]
    
    prompt_lower = prompt.lower()
    for group in kpop_groups:
        if group.lower() in prompt_lower or group in prompt:
            return group
    return None


def gemini_rerank(client: genai.Client, user_prompt: str, k: int, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    schema = {
        "type": "object",
        "properties": {
            "picks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "place_id": {"type": "string"},
                        "rank": {"type": "integer"},
                        "reason": {"type": "string"},
                    },
                    "required": ["place_id", "rank", "reason"],
                },
            }
        },
        "required": ["picks"],
    }

    # 연예인 이름 추출
    celebrity = extract_celebrity_name(user_prompt)
    
    # 프롬프트에 연예인 관련 키워드가 있는지 확인
    has_celebrity_keywords = any(keyword in user_prompt for keyword in [
        "방문", "다녀온", "갔다", "촬영", "브이로그", "vlog", "인스타", "인스타그램"
    ]) or celebrity is not None

    instruction = {
        "task": "후보 장소 목록에서 사용자 프롬프트에 가장 맞는 상위 K개를 고르고 이유를 작성하라.",
        "rules": [
            "반드시 candidates 안에서만 선택(place_id 일치).",
            "rank는 1..K 연속.",
            "reason은 한국어 1~2문장, 프롬프트 조건을 구체적으로 반영."
        ],
        "k": k,
        "user_prompt": user_prompt,
        "candidates": candidates,
    }
    
    # 연예인 관련 검색인 경우 추가 지시
    if has_celebrity_keywords and celebrity:
        instruction["rules"].append(
            f"프롬프트에 '{celebrity}' 관련 내용이 있으므로, 해당 연예인/그룹과 관련성이 높거나 그들이 방문했을 가능성이 있는 장소를 우선적으로 선택하라."
        )
        instruction["rules"].append(
            "장소 이름, 분위기, 스타일이 연예인과 어울리거나 그들이 좋아할 만한 장소를 우선 고려하라."
        )
    elif has_celebrity_keywords:
        instruction["rules"].append(
            "프롬프트에 연예인 관련 키워드가 있으므로, 연예인이 방문했을 가능성이 있거나 그들의 취향과 맞는 감성적인 장소를 우선 선택하라."
        )

    resp = client.models.generate_content(
        model=DEFAULT_MODEL,
        contents=[types.Content(role="user", parts=[types.Part(text=json.dumps(instruction, ensure_ascii=False))])],
        config={"response_mime_type": "application/json", "response_json_schema": schema},
    )
    return json.loads(resp.text)


# ----------------------------
# API 엔드포인트
# ----------------------------
@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


@app.route('/api/config', methods=['GET'])
def get_config():
    """클라이언트 설정 정보 제공 (API 키 등)"""
    google_maps_key = os.environ.get("GOOGLE_MAPS_API_KEY", "")
    return jsonify({
        "google_maps_api_key": google_maps_key
    })


@app.route('/api/location/auto', methods=['GET'])
def get_auto_location():
    """IP 기반 자동 위치 파악"""
    try:
        lat, lon, meta = get_location_by_ip()
        lat, lon = validate_or_fix_latlon(lat, lon)
        return jsonify({
            "success": True,
            "lat": lat,
            "lon": lon,
            "meta": meta
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400


@app.route('/api/recommend', methods=['POST'])
def recommend():
    """장소 추천 API"""
    try:
        data = request.json
        prompt = data.get("prompt", "").strip()
        lat = float(data.get("lat"))
        lon = float(data.get("lon"))
        radius_m = int(data.get("radius_m", DEFAULT_RADIUS_M))
        k = int(data.get("k", DEFAULT_K))
        max_candidates = int(data.get("max_candidates", DEFAULT_MAX_CANDIDATES))
        
        # Places API는 최대 20개까지만 허용
        max_candidates = min(max(1, max_candidates), 20)

        if not prompt:
            return jsonify({"success": False, "error": "프롬프트가 비어있습니다."}), 400

        lat, lon = validate_or_fix_latlon(lat, lon)

        # Gemini client
        gemini_key = os.environ.get("GEMINI_API_KEY")
        if not gemini_key:
            return jsonify({"success": False, "error": "GEMINI_API_KEY 환경변수가 필요합니다."}), 500
        client = genai.Client(api_key=gemini_key)

        # 프롬프트 -> 타입 추출
        included_types = gemini_extract_types(client, prompt)
        if not included_types:
            included_types = ["tourist_attraction"]

        # Places 후보 수집
        raw = places_nearby_search(
            lat=lat,
            lon=lon,
            radius_m=radius_m,
            included_types=included_types,
            max_results=max_candidates,
        )

        places = raw.get("places", [])
        if not places:
            return jsonify({"success": False, "error": "반경 내 후보 장소가 없습니다. 반경(radius_m)을 늘리거나 타입을 바꿔보세요."}), 400

        # 후보 정리 + 거리
        candidates: List[Dict[str, Any]] = []
        for p in places:
            loc = p.get("location", {})
            plat, plon = loc.get("latitude"), loc.get("longitude")
            if plat is None or plon is None:
                continue

            name_obj = p.get("displayName", {})
            name = name_obj.get("text") if isinstance(name_obj, dict) else str(name_obj)

            dist = haversine_km(lat, lon, float(plat), float(plon))

            candidates.append({
                "place_id": p.get("id", ""),
                "name": name or "",
                "address": p.get("formattedAddress"),
                "lat": float(plat),
                "lon": float(plon),
                "distance_km": float(dist),
                "types": p.get("types", []) or [],
                "rating": p.get("rating"),
                "user_ratings_total": p.get("userRatingCount"),
            })

        candidates = sorted(candidates, key=lambda x: x["distance_km"])[:max_candidates]
        k = min(k, len(candidates))

        # Gemini 재랭킹
        reranked = gemini_rerank(client, prompt, k, candidates)
        picks = reranked.get("picks", [])

        cmap = {c["place_id"]: c for c in candidates if c["place_id"]}
        output = []
        used = set()
        for item in sorted(picks, key=lambda x: x.get("rank", 999)):
            pid = item.get("place_id")
            if not pid or pid in used or pid not in cmap:
                continue
            used.add(pid)
            c = cmap[pid]
            output.append({
                "rank": int(item.get("rank", len(output) + 1)),
                "place_id": pid,
                "name": c["name"],
                "address": c.get("address"),
                "lat": c["lat"],
                "lon": c["lon"],
                "distance_km": float(c["distance_km"]),
                "reason": item.get("reason", ""),
                "types": c.get("types", []),
                "rating": c.get("rating"),
                "user_ratings_total": c.get("user_ratings_total"),
            })

        if not output:
            return jsonify({"success": False, "error": "Gemini가 후보 내에서 추천을 구성하지 못했습니다. 프롬프트/반경/타입을 조정해보세요."}), 400

        return jsonify({
            "success": True,
            "included_types": included_types,
            "recommendations": output,
            "user_location": {"lat": lat, "lon": lon},
            "radius_m": radius_m,
            "prompt": prompt
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/recommend-celebrity', methods=['POST'])
def recommend_celebrity():
    """연예인 관련 장소 검색 API (run_pipeline.py 로직 사용)"""
    if not PIPELINE_AVAILABLE:
        return jsonify({
            "success": False,
            "error": "연예인 검색 기능을 사용하려면 pipeline_functions.py 모듈이 필요합니다."
        }), 500
    
    if not CSE_API_KEY or not CSE_CX:
        return jsonify({
            "success": False,
            "error": "GOOGLE_CSE_API_KEY와 GOOGLE_CSE_CX 환경변수가 필요합니다."
        }), 500
    
    try:
        data = request.json
        user_query = normalize_space(data.get("prompt", "").strip())
        lat = float(data.get("lat"))
        lon = float(data.get("lon"))
        max_distance_km = float(data.get("max_distance_km", MAX_DISTANCE_KM))

        if not user_query:
            return jsonify({"success": False, "error": "프롬프트가 비어있습니다."}), 400

        lat, lon = validate_or_fix_latlon(lat, lon)

        # Gemini client
        gemini_key = os.environ.get("GEMINI_API_KEY")
        if not gemini_key:
            return jsonify({"success": False, "error": "GEMINI_API_KEY 환경변수가 필요합니다."}), 500
        client = genai.Client(api_key=gemini_key)

        # 쿼리 의도 분석 (먼저 수행)
        intent = analyze_query_intent(user_query)
        
        # 연예인 이름 추출
        celebrity = guess_celebrity_from_query(user_query)
        if not celebrity:
            return jsonify({"success": False, "error": "프롬프트에서 연예인 이름을 찾을 수 없습니다."}), 400

        # Reverse geocode -> 위치 힌트
        admin = reverse_geocode_admin(lat, lon)
        loc_big = make_location_hint_big(admin)
        
        # 의도에 맞게 max_distance_km 조정
        max_distance_km = min(max_distance_km, intent.max_distance_km)

        # CSE 쿼리 생성 및 검색 (의도 기반 강화)
        query_variants = build_cse_query_variants(user_query=user_query, celebrity=celebrity, loc_big=loc_big, intent=intent)
        all_results = []
        seen_urls = set()
        
        # 의도 기반 쿼리 개수 결정 (대폭 증가)
        if intent.strict_filtering:
            max_queries = 20  # 엄격한 필터링이면 더 많은 쿼리 실행
        elif intent.category != "general":
            max_queries = 18  # 특정 카테고리면 중간
        else:
            max_queries = 15  # 일반이면 기본
        
        # 다중 검색 엔진 병렬 검색 (Google CSE + 네이버 + 네이버 블로그)
        for query in query_variants[:max_queries]:
            try:
                # Google CSE 검색 (결과 수 대폭 증가)
                try:
                    query_results = cse_search(query, total=min(SEARCH_TOTAL_RESULTS, 100), intent=intent)
                    for r in query_results:
                        url = r.get("link", "")
                        if url and url not in seen_urls:
                            seen_urls.add(url)
                            all_results.append(r)
                except Exception:
                    pass
                
                # 네이버 검색 (일반 + 블로그) - Google CSE는 이미 위에서 호출했으므로 제외 (결과 수 대폭 증가)
                try:
                    multi_results = multi_search(query, total_per_source=100, include_google=False)
                    for r in multi_results:
                        url = r.get("link", "")
                        if url and url not in seen_urls:
                            seen_urls.add(url)
                            all_results.append(r)
                except Exception:
                    pass
            except Exception as e:
                continue
        
        if not all_results:
            return jsonify({"success": False, "error": "웹 검색 결과를 찾을 수 없습니다."}), 400

        # 페이지 텍스트 수집 (더 많은 페이지 수집)
        sources = []
        need = min(FETCH_TOP_PAGES, len(all_results))
        
        # 최소한 30개 이상의 소스는 수집하도록 보장
        need = max(need, min(30, len(all_results)))
        
        for r in all_results[:need]:
            url = r.get("link", "")
            if not url:
                continue
            
            page_text = fetch_page_text(url, max_chars=8000)
            if page_text and len(page_text) >= MIN_PAGE_TEXT_CHARS:
                sources.append({
                    "title": r.get("title"),
                    "link": url,
                    "snippet": r.get("snippet"),
                    "page_text": page_text,
                })
        
        # 소스가 부족하면 더 관대한 기준으로 재시도
        if len(sources) < 10:
            # MIN_PAGE_TEXT_CHARS를 낮춰서 더 많은 페이지 수집
            for r in all_results[need:need+30]:
                url = r.get("link", "")
                if not url:
                    continue
                if any(s.get("link") == url for s in sources):
                    continue
                
                page_text = fetch_page_text(url, max_chars=8000)
                if page_text and len(page_text) >= 100:  # 더 낮은 기준
                    sources.append({
                        "title": r.get("title"),
                        "link": url,
                        "snippet": r.get("snippet"),
                        "page_text": page_text,
                    })
                if len(sources) >= 10:
                    break

        if not sources:
            return jsonify({"success": False, "error": "유효한 페이지 텍스트를 수집하지 못했습니다."}), 400

        # Gemini로 장소 추출
        extracted = gemini_extract_places_from_sources(client, user_query, celebrity, loc_big, sources)
        places_raw = extracted.get("places", []) or []

        if not places_raw:
            return jsonify({"success": False, "error": "근거 기반으로 추출된 장소가 없습니다."}), 400

        # 실제 방문 장소와 추천 장소 구분 (더 엄격하게)
        actually_visited = []
        recommended = []
        excluded = []  # 추측성 장소 제외용
        
        for p in places_raw:
            ev_text = p.get("evidence_text", "")
            if not ev_text.strip():
                continue
            
            # 추측성 표현 강력 감지
            ev_lower = ev_text.lower()
            strong_speculation = any(pattern in ev_lower for pattern in [
                "할 법한", "했을 법한", "할 만한", "했을 만한",
                "에 좋은", "에 적합한", "에 완벽한", "와 잘 어울리는",
                "선호하는", "찾을 만한", "찾을"
            ])
            
            # 추측성 표현이 있으면 제외
            if strong_speculation:
                excluded.append(p)
                continue
            
            # 실제 방문 여부 확인
            if is_actually_visited(celebrity, ev_text):
                actually_visited.append(p)
            else:
                # 추천 장소 (실제 방문 근거는 없지만 관련성은 있고 추측성 표현은 없음)
                if is_directly_related(celebrity, ev_text) or not STRICT_EVIDENCE_MATCH:
                    recommended.append(p)
        
        # 실제 방문 장소를 우선 사용, 없으면 추천 장소 사용
        if actually_visited:
            places_raw = actually_visited
        elif recommended:
            places_raw = recommended
        else:
            # 기존 필터 적용 (하위 호환성)
            if STRICT_EVIDENCE_MATCH:
                places_raw = [p for p in places_raw if is_directly_related(celebrity, p.get("evidence_text", ""))]
            else:
                places_raw = [p for p in places_raw if p.get("evidence_text", "").strip()]

        # 지오코딩 (개선된 중복 제거 및 품질 필터링)
        geocoded = []
        geocoded_place_ids = set()
        name_to_places = {}  # 이름 기반 중복 제거
        geocode_failed = []  # 지오코딩 실패한 장소들 (fallback용)
        
        for p in places_raw:
            place_name = normalize_space(p.get("place_name", ""))
            area_hint = normalize_space(p.get("area_hint", ""))
            evidence_text = p.get("evidence_text", "")
            
            # evidence_text에서 위치 정보 추출 (area_hint가 없거나 불명확한 경우)
            if not area_hint or len(area_hint) < 2:
                # evidence_text에서 도시명 추출
                city_keywords = ["서울", "부산", "대구", "인천", "광주", "대전", "울산", "수원", "성남", "고양", "용인", "부천", "안산", "안양", "남양주", "화성", "평택", "의정부", "시흥", "김포", "광명", "군포", "이천", "양주", "오산", "구리", "안성", "포천", "의왕", "하남", "용인", "파주", "이천", "광주", "양평", "동두천", "과천", "가평", "연천", "강남", "강북", "서초", "송파", "마포", "홍대", "이태원", "압구정", "청담", "신사", "한남"]
                for city in city_keywords:
                    if city in evidence_text:
                        area_hint = city
                        break
            
            if not place_name or len(place_name) < 2:
                continue

            g = places_text_search(place_name, area_hint, lat, lon, RADIUS_M)
            if not g:
                # 지오코딩 실패해도 정보는 저장 (fallback용)
                geocode_failed.append({
                    "place_name": place_name,
                    "area_hint": area_hint,
                    "evidence_text": evidence_text,
                    "evidence_url": p.get("evidence_url", ""),
                })
                continue

            place_id = g.get("place_id")
            place_name_normalized = normalize_place_name(g.get("name", ""))
            
            # place_id 기반 중복 제거
            if place_id and place_id in geocoded_place_ids:
                continue
            
            # 이름 기반 중복 제거 (유사한 이름이면 하나만)
            if place_name_normalized in name_to_places:
                existing = name_to_places[place_name_normalized]
                # 기존 것과 거리가 가까우면 (500m 이내) 중복으로 간주
                dist = haversine_km_pipeline(
                    existing["lat"], existing["lon"],
                    g["lat"], g["lon"]
                )
                if dist < 0.5:  # 500m 이내면 중복
                    # 더 많은 근거가 있는 것으로 선택
                    if len(p.get("evidence_text", "")) > len(existing.get("evidence_text", "")):
                        # 기존 것을 교체
                        geocoded.remove(existing)
                        geocoded_place_ids.discard(existing.get("place_id"))
                    else:
                        continue
            
            if place_id:
                geocoded_place_ids.add(place_id)
            
            if place_name_normalized:
                name_to_places[place_name_normalized] = g

            g["evidence_url"] = p.get("evidence_url")
            g["evidence_text"] = p.get("evidence_text")
            g["area_hint"] = area_hint
            g["is_actually_visited"] = is_actually_visited(celebrity, evidence_text)
            geocoded.append(g)

        # 거리 필터링 및 의도 기반 필터링
        final_rows = []
        outside_radius = []
        
        for g in geocoded:
            d = haversine_km_pipeline(lat, lon, g["lat"], g["lon"])
            g["distance_km"] = round(d, 3)
            
            # 의도 기반 필터링 및 점수 계산
            should_include, relevance_score = should_include_place(g, intent)
            g["relevance_score"] = relevance_score
            
            if not should_include:
                continue  # 의도에 맞지 않으면 제외
            
            if d <= max_distance_km:
                final_rows.append(g)
            else:
                outside_radius.append(g)

        # 실제 방문 장소를 최우선으로 정렬하고 최종 10개로 제한
        actually_visited_rows = [g for g in final_rows if g.get("is_actually_visited", False)]
        recommended_rows = [g for g in final_rows if not g.get("is_actually_visited", False)]
        
        # 실제 방문 장소를 먼저, 그 다음 추천 장소
        actually_visited_rows = sorted(actually_visited_rows, key=lambda x: (-x["relevance_score"], x["distance_km"]))
        recommended_rows = sorted(recommended_rows, key=lambda x: (-x["relevance_score"], x["distance_km"]))
        
        # 실제 방문 장소가 있으면 그것을 우선, 없으면 추천 장소 사용 (최대 10개)
        if actually_visited_rows:
            final_rows = actually_visited_rows[:10]
        else:
            final_rows = recommended_rows[:10]
        
        # 의도 기반 추가 필터링 (이미 필터링되었지만 한 번 더 확인)
        final_rows = filter_places_by_intent(final_rows, intent, max_distance_km)
        
        # 결과가 없으면 거리를 자동으로 늘려서 재시도
        if not final_rows and outside_radius:
            # 의도에 맞게 확장 거리 설정
            max_extended_distance = min(intent.max_distance_km * 1.5, 30)
            
            # 거리 순으로 정렬하여 가까운 것부터 포함
            outside_radius_sorted = sorted(outside_radius, key=lambda x: (x["distance_km"], -x["relevance_score"]))
            # 최대 거리까지 확장하여 포함
            for g in outside_radius_sorted:
                if g["distance_km"] <= max_extended_distance and len(final_rows) < 10:
                    # 의도 기반 필터링
                    should_include, relevance_score = should_include_place(g, intent)
                    if not should_include:
                        continue
                    g["relevance_score"] = relevance_score
                    final_rows.append(g)
                elif len(final_rows) >= 10:
                    break
            
            if final_rows:
                max_distance_km = max([g["distance_km"] for g in final_rows])
        
        # 결과가 여전히 없으면 지오코딩된 모든 장소를 거리 순으로 정렬하여 최대 10개 포함
        if not final_rows:
            # 지오코딩된 모든 장소를 거리 순으로 정렬하여 최대 10개 포함
            all_geocoded_sorted = sorted(geocoded, key=lambda x: (
                haversine_km_pipeline(lat, lon, x["lat"], x["lon"]),
                -x.get("relevance_score", 0)
            ))
            for g in all_geocoded_sorted:
                if len(final_rows) >= 10:
                    break
                
                d = haversine_km_pipeline(lat, lon, g["lat"], g["lon"])
                g["distance_km"] = round(d, 3)
                
                # 의도 기반 필터링 (더 관대하게)
                should_include, relevance_score = should_include_place(g, intent)
                # 필터링이 너무 엄격하면 일단 포함 (relevance_score만 낮춤)
                if not should_include:
                    # 의도에 완전히 맞지 않아도 거리가 가까우면 포함
                    if d <= 30:  # 30km 이내면 일단 포함
                        relevance_score = max(0, relevance_score - 20)  # 점수만 낮춤
                    else:
                        continue
                g["relevance_score"] = relevance_score
                
                final_rows.append(g)
            
            # 여전히 결과가 없고 지오코딩 실패한 장소가 있으면, 그것들을 텍스트로라도 반환
            if not final_rows and geocode_failed:
                # 지오코딩 실패한 장소 중에서 실제 방문 근거가 있는 것들을 우선
                failed_with_evidence = [f for f in geocode_failed if is_actually_visited(celebrity, f.get("evidence_text", ""))]
                if failed_with_evidence:
                    # 최대 10개까지 반환 (좌표는 없지만 정보는 제공)
                    for f in failed_with_evidence[:10]:
                        final_rows.append({
                            "name": f.get("place_name", ""),
                            "address": f.get("area_hint", ""),
                            "lat": None,
                            "lon": None,
                            "distance_km": None,
                            "evidence_text": f.get("evidence_text", ""),
                            "evidence_url": f.get("evidence_url", ""),
                            "is_actually_visited": True,
                            "relevance_score": 50,  # 기본 점수
                            "geocode_failed": True,
                        })
                elif geocode_failed:
                    # 실제 방문 근거가 없어도 최소한 정보는 제공
                    for f in geocode_failed[:10]:
                        final_rows.append({
                            "name": f.get("place_name", ""),
                            "address": f.get("area_hint", ""),
                            "lat": None,
                            "lon": None,
                            "distance_km": None,
                            "evidence_text": f.get("evidence_text", ""),
                            "evidence_url": f.get("evidence_url", ""),
                            "is_actually_visited": False,
                            "relevance_score": 30,  # 낮은 점수
                            "geocode_failed": True,
                        })
            
            if final_rows:
                # 거리가 있는 것들만 max_distance_km 계산
                distances = [g["distance_km"] for g in final_rows if g.get("distance_km") is not None]
                if distances:
                    max_distance_km = max(distances)

        
        if not final_rows:
            # 정말로 결과가 없으면 오류
            return jsonify({
                "success": False,
                "error": f"조건을 만족하는 장소를 찾을 수 없습니다.",
                "suggestion": "검색어를 변경하거나 검색 반경을 늘려보세요."
            }), 400

        # 최종 결과를 10개로 제한하고 정렬
        # 실제 방문 장소를 최우선으로, 그 다음 추천 장소
        actually_visited_final = [g for g in final_rows if g.get("is_actually_visited", False)]
        recommended_final = [g for g in final_rows if not g.get("is_actually_visited", False)]
        
        # distance_km가 None인 경우를 처리 (지오코딩 실패한 장소)
        actually_visited_final = sorted(actually_visited_final, key=lambda x: (
            -x.get("relevance_score", 0),
            x.get("distance_km") if x.get("distance_km") is not None else 9999
        ))[:10]
        recommended_final = sorted(recommended_final, key=lambda x: (
            -x.get("relevance_score", 0),
            x.get("distance_km") if x.get("distance_km") is not None else 9999
        ))[:10]
        
        # 실제 방문 장소가 있으면 그것을 우선, 없으면 추천 장소 사용
        if actually_visited_final:
            final_rows = actually_visited_final[:10]
        else:
            final_rows = recommended_final[:10]

        return jsonify({
            "success": True,
            "recommendations": final_rows,
            "user_location": {"lat": lat, "lon": lon},
            "celebrity": celebrity,
            "location_hint": loc_big,
            "max_distance_km": max_distance_km,
            "prompt": user_query
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


def find_free_port(start_port: int = 8080, max_attempts: int = 10) -> int:
    """사용 가능한 포트를 찾는 함수"""
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"{max_attempts}개 포트를 시도했지만 모두 사용 중입니다.")


if __name__ == '__main__':
    # 환경 변수에서 포트 가져오기, 없으면 8080 시도
    requested_port = int(os.environ.get('PORT', 8080))
    
    # 포트가 사용 중이면 자동으로 다른 포트 찾기
    try:
        port = find_free_port(requested_port)
        if port != requested_port:
            print(f"⚠️  포트 {requested_port}이(가) 사용 중입니다. 포트 {port}을(를) 사용합니다.")
    except RuntimeError as e:
        print(f"❌ 오류: {e}")
        sys.exit(1)
    
    print(f"🌐 서버 주소: http://localhost:{port}")
    app.run(host='0.0.0.0', port=port, debug=True)

