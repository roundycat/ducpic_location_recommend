"""
run_pipeline.py의 핵심 함수들을 별도 모듈로 분리
웹 인터페이스에서 재사용 가능하도록 구성
"""
import os
import re
import json
import math
from typing import Any, Dict, List, Optional, Tuple
import requests
from bs4 import BeautifulSoup
from google import genai
from google.genai import types

# 설정 (환경 변수에서 로드)
CSE_API_KEY = os.environ.get("GOOGLE_CSE_API_KEY", "")
CSE_CX = os.environ.get("GOOGLE_CSE_CX", "")
NAVER_CLIENT_ID = os.environ.get("NAVER_CLIENT_ID", "")
NAVER_CLIENT_SECRET = os.environ.get("NAVER_CLIENT_SECRET", "")
PLACES_API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")

SEARCH_TOTAL_RESULTS = int(os.environ.get("SEARCH_TOTAL_RESULTS", "150"))  # 100 -> 150으로 증가
FETCH_TOP_PAGES = int(os.environ.get("FETCH_TOP_PAGES", "50"))  # 30 -> 50으로 증가
MIN_PAGE_TEXT_CHARS = int(os.environ.get("MIN_PAGE_TEXT_CHARS", "200"))
RADIUS_M = int(os.environ.get("RADIUS_M", "20000"))
MAX_DISTANCE_KM = float(os.environ.get("MAX_DISTANCE_KM", "20"))  # 기본값을 20km로 증가
STRICT_EVIDENCE_MATCH = os.environ.get("STRICT_EVIDENCE_MATCH", "false").lower() in ("1", "true", "yes", "y")

PLACES_TEXTSEARCH_URL = "https://places.googleapis.com/v1/places:searchText"
PLACES_FIELD_MASK = (
    "places.id,places.displayName,places.formattedAddress,places.location,"
    "places.types,places.rating,places.userRatingCount"
)
GEOCODE_REVERSE_URL = "https://maps.googleapis.com/maps/api/geocode/json"

BAD_DOMAIN_PATTERNS = ["huggingface.co", "namu.wiki", "namu.moe", "tiktok.com", "daangn.com"]
BAD_EXTENSIONS = (".txt", ".xml", ".json", ".csv", ".zip", ".pdf", ".mp4", ".m3u8")
NEGATIVE_QUERY_KEYWORDS = ["학원", "강사", "레슨", "수강", "프로필", "lecturer", "profile", "dance", "lesson"]


def normalize_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def clean_text(s: str) -> str:
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def is_bad_link(url: str) -> bool:
    if not url:
        return True
    u = url.lower()
    if any(d in u for d in BAD_DOMAIN_PATTERNS):
        return True
    if any(ext in u for ext in BAD_EXTENSIONS):
        return True
    return False


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def guess_celebrity_from_query(user_query: str) -> str:
    uq = normalize_space(user_query)
    if not uq:
        return ""
    return uq.split()[0].strip()


def is_directly_related(celebrity: str, evidence_text: str) -> bool:
    celebrity = (celebrity or "").strip()
    if not celebrity:
        return False
    ev = evidence_text or ""
    if STRICT_EVIDENCE_MATCH:
        if celebrity not in ev:
            return False
    else:
        if not ev or len(ev.strip()) < 10:
            return False
    return True


def is_actually_visited(celebrity: str, evidence_text: str) -> bool:
    """실제 방문 여부 확인 (추측성 표현 강력 제외)"""
    if not evidence_text or not celebrity:
        return False
    
    ev_lower = evidence_text.lower()
    celeb_lower = celebrity.lower()
    
    # 연예인 이름이 포함되어야 함
    if celeb_lower not in ev_lower:
        return False
    
    # 실제 방문/행동 키워드 확인 (더 엄격하게)
    visit_keywords = [
        "방문", "다녀온", "갔다", "갔던", "가본", "가봤", "가봤던",
        "촬영", "촬영지", "촬영한", "촬영했던", "촬영했다",
        "인스타", "인스타그램", "게시", "포스팅", "올린", "올렸",
        "브이로그", "vlog", "vlog에서",
        "공연", "출연", "참석", "참가",
        "먹었", "마셨", "사진", "사진을", "사진에", "찍었",
        "체험", "이용", "예약", "리뷰", "후기"
    ]
    
    # 강력한 추측성 표현 감지 (이것들이 있으면 무조건 False)
    strong_speculation_patterns = [
        "할 법한", "했을 법한", "할 만한", "했을 만한",
        "에 좋은", "에 적합한", "에 완벽한", "에 이상적인",
        "와 잘 어울리는", "와 어울리는", "와 잘 맞는",
        "느낌", "분위기", "컨셉", "이미지",
        "추천", "추천할", "좋을", "완벽한", "이상적인",
        "연상", "연상시킨다", "같다", "비슷하다",
        "선호하는", "선호할", "찾을 만한", "찾을"
    ]
    
    # 강력한 추측성 표현이 있으면 무조건 False
    if any(pattern in ev_lower for pattern in strong_speculation_patterns):
        return False
    
    # 실제 방문 키워드가 반드시 있어야 함 (없으면 False)
    if not any(kw in ev_lower for kw in visit_keywords):
        return False
    
    # 실제 방문 키워드가 있으면 True
    return True


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
    parts = []
    for k in ["province", "city", "district"]:
        v = (admin.get(k) or "").strip()
        if v and v not in parts:
            parts.append(v)
    return " ".join(parts).strip()


def build_cse_query_variants(user_query: str, celebrity: str, loc_big: str, intent: Optional[Any] = None) -> List[str]:
    """강화된 쿼리 변형 생성 - 의도 기반 스마트한 조합"""
    celeb = f'"{celebrity}"'
    variants = []
    
    neg_sites = " ".join([f"-site:{d}" for d in BAD_DOMAIN_PATTERNS])
    neg_kw = " ".join([f"-{k}" for k in NEGATIVE_QUERY_KEYWORDS])
    
    # 의도 기반 키워드 추출
    intent_keywords = []
    if intent and hasattr(intent, 'category'):
        if intent.category == "restaurant":
            intent_keywords = ["맛집", "식당", "레스토랑", "음식점", "식사", "요리"]
        elif intent.category == "cafe":
            intent_keywords = ["카페", "커피", "브런치"]
        elif intent.category == "emotional_space":
            intent_keywords = ["감성", "공간", "분위기", "예쁜", "아름다운", "로맨틱", "힐링"]
        elif intent.category == "tourist_attraction":
            intent_keywords = ["관광지", "명소", "여행지", "데이트"]
    
    # 1. 기본 패턴들 (위치 포함)
    if loc_big:
        simple = f'{celeb} {loc_big}'
        variants.append(normalize_space(f"{simple} {neg_sites} {neg_kw}"))
    
    # 2. 연예인명만 (가장 넓은 검색)
    variants.append(normalize_space(f"{celeb} {neg_sites} {neg_kw}"))
    
    # 3. 의도 기반 패턴 우선 생성
    if intent_keywords:
        for keyword in intent_keywords[:3]:  # 상위 3개만 사용
            pattern = f'{celeb} {keyword}'
            if loc_big:
                variants.insert(len(variants), normalize_space(f"{pattern} {loc_big} {neg_sites} {neg_kw}"))
            variants.append(normalize_space(f"{pattern} {neg_sites} {neg_kw}"))
    
    # 4. 확장된 패턴들 (더 많은 키워드 조합)
    base_patterns = [
        f'{celeb} 촬영지',
        f'{celeb} 방문',
        f'{celeb} 다녀온',
        f'{celeb} 갔다',
        f'{celeb} 가본',
        f'{celeb} 갔던',
        f'{celeb} 브이로그',
        f'{celeb} vlog',
        f'{celeb} 인스타',
        f'{celeb} 인스타그램',
        f'{celeb} 포스팅',
        f'{celeb} 게시',
        f'{celeb} 추천',
    ]
    
    # 의도에 맞는 패턴만 추가
    if not intent_keywords or any(kw in ["맛집", "식당", "레스토랑"] for kw in intent_keywords):
        base_patterns.extend([f'{celeb} 맛집', f'{celeb} 식당', f'{celeb} 레스토랑'])
    if not intent_keywords or "카페" in intent_keywords:
        base_patterns.append(f'{celeb} 카페')
    if not intent_keywords or any(kw in ["관광지", "명소", "여행지"] for kw in intent_keywords):
        base_patterns.extend([f'{celeb} 관광지', f'{celeb} 명소', f'{celeb} 여행지', f'{celeb} 데이트'])
    
    for pattern in base_patterns:
        core = pattern
        if loc_big:
            core += f" {loc_big}"
        variants.append(normalize_space(f"{core} {neg_sites} {neg_kw}"))
    
    # 5. 사용자 쿼리 기반 변형 (강화된 버전)
    if user_query and user_query != celebrity:
        user_q = normalize_space(user_query)
        # 사용자 쿼리에서 키워드 추출
        user_keywords = []
        extended_keywords = ["감성", "공간", "카페", "맛집", "관광지", "명소", "여행지", 
                           "분위기", "예쁜", "아름다운", "로맨틱", "힐링", "데이트", 
                           "브런치", "갤러리", "전시", "문화", "책방"]
        
        for keyword in extended_keywords:
            if keyword in user_q:
                user_keywords.append(keyword)
        
        # 사용자 쿼리 전체 + 위치 (최우선)
        if loc_big:
            variants.insert(0, normalize_space(f"{user_q} {loc_big} {neg_sites} {neg_kw}"))
        variants.insert(0, normalize_space(f"{user_q} {neg_sites} {neg_kw}"))
        
        # 연예인 + 사용자 쿼리 전체
        if loc_big:
            variants.insert(1, normalize_space(f"{celeb} {user_q} {loc_big} {neg_sites} {neg_kw}"))
        variants.insert(1, normalize_space(f"{celeb} {user_q} {neg_sites} {neg_kw}"))
        
        # 연예인 + 사용자 키워드 조합 (감성 공간 같은 조합)
        for keyword in user_keywords:
            combo = f'{celeb} {keyword}'
            if loc_big:
                variants.append(normalize_space(f"{combo} {loc_big} {neg_sites} {neg_kw}"))
            variants.append(normalize_space(f"{combo} {neg_sites} {neg_kw}"))
        
        # 특수 조합 (의도 기반)
        if intent_keywords:
            # 감성 공간 조합
            if "감성" in user_keywords or "공간" in user_keywords:
                variants.insert(2, normalize_space(f'{celeb} 감성 공간 {loc_big if loc_big else ""} {neg_sites} {neg_kw}'))
                variants.insert(2, normalize_space(f'{celeb} 감성 카페 {loc_big if loc_big else ""} {neg_sites} {neg_kw}'))
                variants.insert(2, normalize_space(f'{celeb} 감성 갤러리 {loc_big if loc_big else ""} {neg_sites} {neg_kw}'))
            
            # 맛집 조합
            if any(kw in user_keywords for kw in ["맛집", "식당"]):
                variants.insert(2, normalize_space(f'{celeb} 맛집 추천 {loc_big if loc_big else ""} {neg_sites} {neg_kw}'))
                variants.insert(2, normalize_space(f'{celeb} 맛집 방문 {loc_big if loc_big else ""} {neg_sites} {neg_kw}'))
        
        # 동의어 확장 쿼리
        for keyword in user_keywords[:2]:  # 상위 2개만
            # 동의어 사전 활용
            synonyms = {
                "맛집": ["식당", "레스토랑", "음식점"],
                "카페": ["커피", "브런치"],
                "감성": ["분위기", "로맨틱", "힐링"],
                "공간": ["장소", "스페이스"]
            }
            if keyword in synonyms:
                for syn in synonyms[keyword][:2]:  # 각 동의어당 2개만
                    combo = f'{celeb} {syn}'
                    if loc_big:
                        variants.append(normalize_space(f"{combo} {loc_big} {neg_sites} {neg_kw}"))
                    variants.append(normalize_space(f"{combo} {neg_sites} {neg_kw}"))
    
    # 6. 영어/한글 혼용 검색 (연예인명이 한글인 경우)
    if any(ord(char) >= 0xAC00 and ord(char) <= 0xD7A3 for char in celebrity):
        # 한글 연예인명의 경우 영어 검색도 시도
        celeb_eng = celebrity.replace(" ", "").lower()
        if celeb_eng != celebrity.lower():
            variants.append(normalize_space(f'"{celeb_eng}" {loc_big if loc_big else ""} {neg_sites} {neg_kw}'))
    
    # 중복 제거 및 우선순위 정렬
    variants = list(dict.fromkeys(variants))  # 중복 제거
    
    # 우선순위 정렬: 사용자 쿼리 포함 > 의도 키워드 포함 > 기본 패턴
    def query_priority(q: str) -> int:
        priority = 0
        if user_query and user_query != celebrity and user_query.lower() in q.lower():
            priority += 100  # 사용자 쿼리 포함 최우선
        if intent_keywords and any(kw in q for kw in intent_keywords):
            priority += 50  # 의도 키워드 포함
        if loc_big and loc_big in q:
            priority += 20  # 위치 포함
        if "방문" in q or "다녀온" in q or "갔다" in q:
            priority += 10  # 방문 관련 키워드
        return priority
    
    variants = sorted(variants, key=query_priority, reverse=True)
    
    return variants


def cse_search(query: str, total: int = 50, intent: Optional[Any] = None) -> List[Dict[str, Any]]:
    if not CSE_API_KEY or not CSE_CX:
        raise RuntimeError("GOOGLE_CSE_API_KEY, GOOGLE_CSE_CX 환경변수가 필요합니다.")

    # Google CSE는 최대 100개까지 가능 (한 번에 10개씩)
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


def naver_search(query: str, search_type: str = "webkr", total: int = 50) -> List[Dict[str, Any]]:
    """네이버 검색 API (일반 검색 또는 블로그 검색)"""
    if not NAVER_CLIENT_ID or not NAVER_CLIENT_SECRET:
        print(f"[네이버 검색] API 키가 설정되지 않았습니다. (query: {query})")
        return []  # API 키가 없으면 빈 리스트 반환
    
    # search_type: "webkr" (일반 검색) 또는 "blog" (블로그 검색)
    url = f"https://openapi.naver.com/v1/search/{search_type}"
    headers = {
        "X-Naver-Client-Id": NAVER_CLIENT_ID,
        "X-Naver-Client-Secret": NAVER_CLIENT_SECRET
    }
    
    total = max(1, min(total, 100))  # 네이버 API는 최대 100개
    results: List[Dict[str, Any]] = []
    start = 1
    
    try:
        while len(results) < total:
            display = min(100, total - len(results))  # 네이버는 최대 100개씩
            params = {
                "query": query,
                "display": display,
                "start": start,
                "sort": "sim"  # sim: 정확도순, date: 날짜순
            }
            
            try:
                r = requests.get(url, headers=headers, params=params, timeout=15)
                if r.status_code >= 400:
                    error_data = r.json() if r.content else {}
                    error_msg = error_data.get("errorMessage", r.text[:200])
                    print(f"[네이버 검색] API 오류 (status {r.status_code}): {error_msg} (query: {query})")
                    break  # 오류 발생 시 중단
                
                data = r.json()
                items = data.get("items", []) or []
                if not items:
                    break
                
                for it in items:
                    link = it.get("link") or it.get("bloggerlink") or ""
                    if not link or is_bad_link(link):
                        continue
                    
                    # 네이버 검색 결과 형식 통일
                    title = it.get("title", "").replace("<b>", "").replace("</b>", "")
                    snippet = it.get("description", "").replace("<b>", "").replace("</b>", "")
                    
                    results.append({
                        "title": title,
                        "link": link,
                        "snippet": snippet
                    })
                
                start += len(items)
                if len(items) < display:
                    break
            except Exception:
                break  # 내부 try-except에서 예외 발생 시 루프 종료
        
        if results:
            print(f"[네이버 검색] {search_type} 성공: {len(results)}개 결과 (query: {query})")
        else:
            print(f"[네이버 검색] {search_type} 결과 없음 (query: {query})")
    except Exception as e:
        print(f"[네이버 검색] 예외 발생: {str(e)} (query: {query})")
    
    return results


def naver_blog_search(query: str, total: int = 50) -> List[Dict[str, Any]]:
    """네이버 블로그 검색 API"""
    return naver_search(query, search_type="blog", total=total)


def multi_search(query: str, total_per_source: int = 30, include_google: bool = False) -> List[Dict[str, Any]]:
    """다중 검색 엔진 병렬 검색 (네이버 + 네이버 블로그, 선택적으로 Google CSE 포함)"""
    all_results = []
    seen_urls = set()
    
    # 1. Google CSE 검색 (선택적)
    if include_google:
        try:
            if CSE_API_KEY and CSE_CX:
                cse_results = cse_search(query, total=total_per_source)
                for r in cse_results:
                    url = r.get("link", "")
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        all_results.append(r)
        except Exception:
            pass
    
    # 2. 네이버 일반 검색
    try:
        if NAVER_CLIENT_ID and NAVER_CLIENT_SECRET:
            naver_results = naver_search(query, search_type="webkr", total=total_per_source)
            for r in naver_results:
                url = r.get("link", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    all_results.append(r)
        else:
            print(f"[multi_search] 네이버 API 키가 없어 네이버 검색을 건너뜁니다.")
    except Exception as e:
        print(f"[multi_search] 네이버 일반 검색 오류: {str(e)}")
    
    # 3. 네이버 블로그 검색
    try:
        if NAVER_CLIENT_ID and NAVER_CLIENT_SECRET:
            blog_results = naver_blog_search(query, total=total_per_source)
            for r in blog_results:
                url = r.get("link", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    all_results.append(r)
    except Exception as e:
        print(f"[multi_search] 네이버 블로그 검색 오류: {str(e)}")
    
    return all_results


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

    # 사용자 쿼리에서 키워드 추출 및 우선순위 설정
    user_keywords = []
    priority_keywords = []
    exclude_keywords = []
    
    query_lower = user_query.lower()
    
    # 감성/분위기 관련 키워드
    if any(kw in query_lower for kw in ["감성", "분위기", "예쁜", "아름다운", "로맨틱", "힐링"]):
        priority_keywords.extend(["카페", "갤러리", "전시관", "미술관", "문화공간", "책방", "브런치", "와인바"])
        exclude_keywords.extend(["체육관", "운동장", "경기장", "공연장", "콘서트홀"])
    
    # 공간/장소 관련 키워드
    if "공간" in query_lower or "장소" in query_lower:
        priority_keywords.extend(["카페", "스튜디오", "갤러리", "전시관", "문화센터", "복합문화공간"])
    
    # 카페 관련
    if "카페" in query_lower:
        priority_keywords.append("카페")
    
    # 맛집 관련
    if any(kw in query_lower for kw in ["맛집", "식당", "레스토랑"]):
        priority_keywords.extend(["맛집", "식당", "레스토랑", "브런치"])
    
    instruction = {
        "role": "너는 웹 페이지에서 장소를 추출하는 전문 정보추출기다. 매우 적극적이고 정확하게 추출해야 한다.",
        "task": f"아래 sources에서 '{celebrity}'와 관련된 모든 장소를 가능한 한 많이 추출하라.",
        "rules": [
            "근거가 sources에 실제로 존재할 때만 추출(추측 절대 금지).",
            "evidence_text는 반드시 연예인이 실제로 방문했거나 관련된 구체적인 사실만 사용 (인용 가능한 형태).",
            "추측성 표현은 절대 사용하지 말 것: '~할 법한', '~했을 법한', '~에 좋은', '~에 적합한', '~와 잘 어울리는', '~할 만한', '~했을 만한', '~선호하는', '~찾을 만한' 등.",
            "evidence_text에는 반드시 '방문', '다녀온', '갔다', '촬영', '인스타그램', '브이로그', '게시', '올린', '찍었' 같은 실제 행동이 포함되어야 함.",
            "만약 sources에 '~할 법한', '~에 좋은' 같은 추측성 표현만 있고 실제 방문 근거가 없으면 절대 추출하지 말 것.",
            "place_name은 장소명만 간결하고 정확하게 추출 (브랜드명, 지점명 포함 가능).",
            "area_hint는 반드시 evidence_text나 sources에서 언급된 실제 위치 정보를 추출하라 (예: '서울', '수원', '강남', '홍대' 등).",
            "area_hint는 'OO시/OO구' 또는 '서울', '부산' 같은 도시명까지 추출 (없으면 추측해도 OK, 하지만 가능하면 실제 언급된 위치 사용).",
            "같은 이름의 장소가 여러 지역에 있을 수 있으므로, evidence_text에서 언급된 정확한 위치를 area_hint에 반드시 포함하라.",
            "모든 종류의 장소를 추출하되, 사용자 쿼리의 의도를 반영하라.",
            f"사용자 쿼리: '{user_query}'",
            f"우선 추출할 장소 유형: {', '.join(priority_keywords) if priority_keywords else '카페/맛집/식당/레스토랑/관광지/명소/공원/전시/박물관/미술관/갤러리/쇼핑몰/영화관/공연장/리조트/호텔/펜션/게스트하우스/스튜디오/촬영지'}",
            f"가능하면 제외할 장소 유형: {', '.join(exclude_keywords) if exclude_keywords else '학원/레슨/프로필/강사/개인사업'}",
            "학원/레슨/프로필/강사/개인사업 같은 성격이면 제외.",
            "같은 장소가 여러 번 언급되면 모두 추출해도 OK (중복 제거는 나중에).",
            "장소명이 부분적으로만 나와도 추출 (예: '스타벅스'만 나와도 OK, 나중에 지오코딩으로 보정).",
            "인스타그램, 블로그, 뉴스 등 다양한 출처에서 언급된 장소를 모두 추출.",
            "반드시 최소 80개 이상 추출해야 함 (가능하면 100개 이상).",
            "장소명이 불명확하거나 부분적이어도 추출 시도 (예: '그 카페', '그곳' 같은 표현도 문맥상 장소명을 추론 가능하면 추출).",
            "category_hint에 장소 유형을 명시하라 (예: 카페, 갤러리, 전시관, 맛집 등).",
        ],
        "celebrity": celebrity,
        "user_query": user_query,
        "user_location_hint": location_hint,
        "sources": sources,
        "important": f"사용자가 '{user_query}'로 검색했으므로, 이 의도에 맞는 장소를 우선적으로 추출하세요. {'감성적이고 분위기 있는' if '감성' in query_lower else ''} {'작고 아늑한' if '공간' in query_lower else ''} 장소를 중점적으로 찾으세요."
    }

    resp = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[types.Content(role="user", parts=[types.Part(text=json.dumps(instruction, ensure_ascii=False))])],
        config={"response_mime_type": "application/json", "response_json_schema": schema},
    )
    return json.loads(resp.text)


def places_text_search(place_name: str, area_hint: str, bias_lat: float, bias_lon: float, radius_m: int) -> Optional[Dict[str, Any]]:
    """개선된 지오코딩 전략 - 더 많은 쿼리 변형과 스마트한 매칭"""
    queries = []
    
    # 1. 원본 장소명 (최우선)
    queries.append(place_name)
    
    # 2. 위치 힌트 포함 쿼리들
    if area_hint:
        queries.append(f"{place_name} {area_hint}")
        area_parts = area_hint.split()
        if len(area_parts) > 1:
            # 첫 번째 부분만 (예: "경기도 수원시" -> "경기도")
            queries.append(f"{place_name} {area_parts[0]}")
            # 마지막 부분만 (예: "경기도 수원시" -> "수원시")
            queries.append(f"{place_name} {area_parts[-1]}")
    
    # 3. 장소명 정리 (괄호, 특수문자 제거)
    clean_name = re.sub(r'[\(\)\[\]【】「」『』]', '', place_name).strip()
    if clean_name and clean_name != place_name:
        queries.append(clean_name)
        if area_hint:
            queries.append(f"{clean_name} {area_hint}")
    
    # 4. 공백 제거 버전 (예: "스타 벅스" -> "스타벅스")
    no_space = place_name.replace(" ", "").replace("　", "")
    if no_space != place_name:
        queries.append(no_space)
        if area_hint:
            queries.append(f"{no_space} {area_hint}")
    
    # 5. 불필요한 단어 제거 (더 많은 패턴)
    remove_words = [
        "카페", "식당", "맛집", "레스토랑", "공원", "박물관", "미술관",
        "갤러리", "전시관", "문화센터", "체험관", "쇼핑몰", "마트",
        "영화관", "극장", "공연장", "리조트", "호텔", "펜션"
    ]
    for word in remove_words:
        if word in place_name:
            simplified = place_name.replace(word, "").strip()
            if simplified and len(simplified) >= 2:  # 최소 2글자는 남겨야 함
                queries.append(simplified)
                if area_hint:
                    queries.append(f"{simplified} {area_hint}")
            break
    
    # 6. 지점명 제거 (예: "스타벅스 강남점" -> "스타벅스")
    branch_patterns = [r'\s*(점|지점|본점|직영점|체인점)', r'\s*\([^)]*점[^)]*\)', r'\s*\[[^\]]*점[^\]]*\]']
    for pattern in branch_patterns:
        branch_removed = re.sub(pattern, '', place_name, flags=re.IGNORECASE).strip()
        if branch_removed and branch_removed != place_name:
            queries.append(branch_removed)
            if area_hint:
                queries.append(f"{branch_removed} {area_hint}")
    
    # 7. 영어/한글 혼용 처리
    if any(ord(char) >= 0xAC00 and ord(char) <= 0xD7A3 for char in place_name):
        # 한글이 포함된 경우 영어만 추출 시도
        eng_only = re.sub(r'[가-힣\s]', '', place_name).strip()
        if eng_only and len(eng_only) >= 2:
            queries.append(eng_only)
            if area_hint:
                queries.append(f"{eng_only} {area_hint}")
    
    # 8. 부분 매칭 (장소명이 긴 경우 앞부분만)
    if len(place_name) > 10:
        first_part = place_name[:10].strip()
        queries.append(first_part)
        if area_hint:
            queries.append(f"{first_part} {area_hint}")
    
    # 중복 제거 (순서 유지)
    queries = list(dict.fromkeys(queries))

    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": PLACES_API_KEY,
        "X-Goog-FieldMask": PLACES_FIELD_MASK,
    }

    # 쿼리 우선순위 정렬 (위치 힌트가 있는 쿼리를 최우선으로)
    def query_priority(q: str) -> int:
        priority = 0
        if area_hint and area_hint in q:
            priority += 50  # 위치 힌트 포함 시 최우선 (대폭 증가)
        if len(q) > len(place_name):
            priority += 5  # 더 긴 쿼리 (더 구체적)
        if q == place_name:
            priority += 20  # 원본 장소명
        return priority
    
    queries = sorted(queries, key=query_priority, reverse=True)
    
    # 여러 결과를 수집하여 가장 적합한 것 선택
    candidates = []
    
    for q in queries:
        for attempt_radius in [radius_m, radius_m * 2, radius_m * 4]:
            body = {
                "textQuery": q,
                "maxResultCount": 3,  # 여러 결과 확인
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

                for p in places:
                    loc = p.get("location", {})
                    lat = loc.get("latitude")
                    lon = loc.get("longitude")
                    name_obj = p.get("displayName", {})
                    name = name_obj.get("text") if isinstance(name_obj, dict) else str(name_obj)

                    if lat is None or lon is None:
                        continue
                    
                    # 거리 계산
                    dist = haversine_km(bias_lat, bias_lon, float(lat), float(lon))
                    
                    # 이름 유사도 계산
                    name_similarity = calculate_name_similarity(place_name, name)
                    
                    # area_hint와 주소 매칭 점수 계산
                    location_match_score = 0.0
                    address = p.get("formattedAddress", "").lower()
                    if area_hint:
                        area_lower = area_hint.lower()
                        # area_hint가 주소에 포함되어 있으면 높은 점수
                        if area_lower in address:
                            location_match_score = 1.0
                        else:
                            # 부분 매칭 (예: "서울" -> "서울시", "수원" -> "수원시")
                            area_parts = area_lower.split()
                            for part in area_parts:
                                if part in address:
                                    location_match_score += 0.5
                            location_match_score = min(location_match_score, 1.0)
                    
                    candidates.append({
                        "place_id": p.get("id"),
                        "name": name,
                        "address": p.get("formattedAddress"),
                        "lat": float(lat),
                        "lon": float(lon),
                        "types": p.get("types", []) or [],
                        "rating": p.get("rating"),
                        "user_ratings_total": p.get("userRatingCount"),
                        "distance": dist,
                        "name_similarity": name_similarity,
                        "location_match_score": location_match_score,
                        "query_used": q,
                    })
                    
                    # area_hint와 일치하고 이름도 유사하면 즉시 반환
                    if area_hint and location_match_score > 0.5 and name_similarity > 0.7:
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
                    
                    # 좋은 매칭을 찾으면 즉시 반환
                    if name_similarity > 0.8 and dist < radius_m / 1000:  # 거리도 가까우면
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
    
    # 후보가 있으면 가장 적합한 것 선택
    if candidates:
        # area_hint가 있으면 위치 매칭 점수를 최우선으로 고려
        for c in candidates:
            if area_hint:
                # 위치 매칭(40%) + 이름 유사도(40%) + 거리(20%)
                score = (
                    c.get("location_match_score", 0) * 0.4 +
                    c["name_similarity"] * 0.4 +
                    (1 - min(c["distance"] / 50, 1)) * 0.2
                )
            else:
                # 이름 유사도(70%) + 거리(30%)
                score = c["name_similarity"] * 0.7 + (1 - min(c["distance"] / 50, 1)) * 0.3
            c["score"] = score
        
        # 점수로 정렬하여 최적의 매칭 선택
        best = max(candidates, key=lambda x: x["score"])
        
        # area_hint가 있고 위치가 일치하지 않으면, 일치하는 것이 있는지 다시 확인
        if area_hint and best.get("location_match_score", 0) < 0.5:
            location_matched = [c for c in candidates if c.get("location_match_score", 0) > 0.5]
            if location_matched:
                # 위치가 일치하는 것 중에서 가장 좋은 것 선택
                best = max(location_matched, key=lambda x: x["name_similarity"] * 0.6 + (1 - min(x["distance"] / 50, 1)) * 0.4)
        
        return {
            "place_id": best["place_id"],
            "name": best["name"],
            "address": best["address"],
            "lat": best["lat"],
            "lon": best["lon"],
            "types": best["types"],
            "rating": best["rating"],
            "user_ratings_total": best["user_ratings_total"],
        }

    return None


def calculate_name_similarity(name1: str, name2: str) -> float:
    """간단한 이름 유사도 계산 (0.0 ~ 1.0)"""
    if not name1 or not name2:
        return 0.0
    
    name1_lower = name1.lower().strip()
    name2_lower = name2.lower().strip()
    
    # 완전 일치
    if name1_lower == name2_lower:
        return 1.0
    
    # 한쪽이 다른 쪽을 포함
    if name1_lower in name2_lower or name2_lower in name1_lower:
        return 0.8
    
    # 공통 부분 계산
    common_chars = set(name1_lower) & set(name2_lower)
    if not common_chars:
        return 0.0
    
    # 간단한 유사도 (공통 문자 비율)
    max_len = max(len(name1_lower), len(name2_lower))
    similarity = len(common_chars) / max_len if max_len > 0 else 0.0
    
    return min(similarity * 1.2, 1.0)  # 약간 보정

