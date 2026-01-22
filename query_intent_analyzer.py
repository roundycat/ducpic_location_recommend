"""
사용자 쿼리 의도 분석 모듈
쿼리에서 의도를 추출하고 필터링 규칙을 생성
"""
from typing import Dict, List, Set, Tuple, Any, Optional


# 동의어 및 유사어 사전
SYNONYM_DICT = {
    "맛집": ["식당", "레스토랑", "음식점", "식사", "요리", "맛", "음식"],
    "식당": ["맛집", "레스토랑", "음식점", "식사", "요리"],
    "카페": ["커피", "브런치", "다방", "카페테리아"],
    "감성": ["분위기", "예쁜", "아름다운", "로맨틱", "힐링", "감성적"],
    "공간": ["장소", "스페이스", "플레이스"],
    "관광지": ["명소", "여행지", "데이트", "스팟"],
    "방문": ["다녀온", "갔다", "가본", "갔던", "찾아간", "찾아온"],
    "촬영": ["촬영지", "촬영한", "촬영했던", "사진", "영상"],
    "인스타": ["인스타그램", "포스팅", "게시", "올린"],
    "브이로그": ["vlog", "브이로그", "비디오", "영상"]
}


def expand_keywords(keyword: str) -> List[str]:
    """키워드 확장 (동의어 포함)"""
    expanded = [keyword]
    if keyword in SYNONYM_DICT:
        expanded.extend(SYNONYM_DICT[keyword])
    return expanded


class QueryIntent:
    """쿼리 의도 정보"""
    def __init__(self):
        self.category: str = "general"  # general, restaurant, cafe, emotional_space, etc.
        self.required_types: Set[str] = set()  # 필수 장소 타입
        self.excluded_types: Set[str] = set()  # 제외할 장소 타입
        self.excluded_keywords: Set[str] = set()  # 제외할 이름 키워드
        self.priority_types: Set[str] = set()  # 우선할 장소 타입
        self.max_distance_km: float = 20.0  # 최대 거리
        self.strict_filtering: bool = False  # 엄격한 필터링 여부


def analyze_query_intent(user_query: str) -> QueryIntent:
    """사용자 쿼리에서 의도를 분석하고 필터링 규칙 생성"""
    intent = QueryIntent()
    query_lower = user_query.lower()
    
    # 1. 맛집/식당 의도
    if any(kw in query_lower for kw in ["맛집", "식당", "레스토랑", "음식점", "식사"]):
        intent.category = "restaurant"
        intent.required_types = {"restaurant", "food", "meal_takeaway", "cafe", "bakery"}
        intent.excluded_types = {
            "stadium", "gym", "sports_complex", "convention_center",
            "theater", "performing_arts_theater", "movie_theater",
            "university", "school", "broadcasting_station", "hospital",
            "amusement_park", "zoo", "park"
        }
        intent.excluded_keywords = {
            "방송", "MBC", "KBS", "SBS", "대학교", "대학", "학교",
            "병원", "체육관", "운동장", "경기장", "공연장", "극장"
        }
        intent.priority_types = {"restaurant", "food", "meal_takeaway"}
        intent.strict_filtering = True
        intent.max_distance_km = 25.0
    
    # 2. 카페 의도
    elif "카페" in query_lower:
        intent.category = "cafe"
        intent.required_types = {"cafe", "bakery", "restaurant"}
        intent.excluded_types = {
            "stadium", "gym", "sports_complex", "convention_center",
            "theater", "performing_arts_theater", "movie_theater",
            "university", "school", "broadcasting_station"
        }
        intent.excluded_keywords = {
            "방송", "MBC", "KBS", "SBS", "대학교", "대학", "학교",
            "체육관", "운동장", "경기장", "공연장", "극장"
        }
        intent.priority_types = {"cafe", "bakery"}
        intent.strict_filtering = True
    
    # 3. 감성 공간 의도
    elif any(kw in query_lower for kw in ["감성", "공간", "분위기", "예쁜", "아름다운", "로맨틱", "힐링"]):
        intent.category = "emotional_space"
        intent.required_types = {
            "cafe", "art_gallery", "museum", "book_store", 
            "restaurant", "tourist_attraction", "park"
        }
        intent.excluded_types = {
            "stadium", "gym", "sports_complex", "convention_center",
            "theater", "performing_arts_theater", "movie_theater",
            "university", "school", "broadcasting_station",
            "amusement_park", "zoo", "hospital"
        }
        intent.excluded_keywords = {
            "극장", "공연장", "체육관", "운동장", "경기장", 
            "국립극장", "예술의전당", "올림픽공원",
            "방송", "MBC", "KBS", "SBS", "대학교", "대학", "학교"
        }
        intent.priority_types = {
            "cafe", "art_gallery", "museum", "book_store", 
            "restaurant", "tourist_attraction"
        }
        intent.strict_filtering = True
    
    # 4. 관광지/명소 의도
    elif any(kw in query_lower for kw in ["관광지", "명소", "여행지", "데이트"]):
        intent.category = "tourist_attraction"
        intent.required_types = {
            "tourist_attraction", "park", "museum", "art_gallery",
            "cafe", "restaurant", "shopping_mall"
        }
        intent.excluded_types = {
            "stadium", "gym", "sports_complex", "convention_center",
            "university", "school", "broadcasting_station", "hospital"
        }
        intent.excluded_keywords = {
            "방송", "MBC", "KBS", "SBS", "대학교", "대학", "학교", "병원"
        }
        intent.priority_types = {
            "tourist_attraction", "park", "museum", "art_gallery"
        }
    
    # 5. 일반 (기본값)
    else:
        intent.category = "general"
        intent.excluded_types = {
            "hospital", "funeral_home", "cemetery"
        }
        intent.excluded_keywords = {
            "병원", "장례식장", "묘지"
        }
    
    return intent


def get_query_keywords_for_search(intent: QueryIntent, user_query: str) -> List[str]:
    """검색 쿼리 생성을 위한 키워드 추출"""
    keywords = []
    query_lower = user_query.lower()
    
    if intent.category == "restaurant":
        keywords.extend(["맛집", "식당", "레스토랑", "음식점", "식사", "요리"])
    elif intent.category == "cafe":
        keywords.extend(["카페", "커피", "브런치"])
    elif intent.category == "emotional_space":
        keywords.extend(["감성", "공간", "분위기", "예쁜", "아름다운", "로맨틱", "힐링"])
    elif intent.category == "tourist_attraction":
        keywords.extend(["관광지", "명소", "여행지", "데이트"])
    
    # 사용자 쿼리에서 직접 추출
    for keyword in keywords:
        if keyword in query_lower:
            keywords.extend(expand_keywords(keyword))
    
    return list(set(keywords))  # 중복 제거


def should_include_place(place: Dict[str, Any], intent: QueryIntent) -> Tuple[bool, float]:
    """
    장소가 의도에 맞는지 확인하고 관련도 점수 반환
    Returns: (should_include, relevance_score)
    """
    place_types = set(place.get("types", []) or [])
    place_name = (place.get("name") or "").lower()
    
    score = 1.0
    
    # 1. 필수 타입 체크 (엄격한 필터링)
    if intent.strict_filtering and intent.required_types:
        if not any(t in place_types for t in intent.required_types):
            return (False, -10.0)  # 필수 타입이 없으면 제외
    
    # 2. 제외 타입 체크
    if any(t in place_types for t in intent.excluded_types):
        return (False, -10.0)  # 제외 타입이 있으면 제외
    
    # 3. 제외 키워드 체크
    if any(kw in place_name for kw in intent.excluded_keywords):
        return (False, -10.0)  # 제외 키워드가 있으면 제외
    
    # 4. 우선 타입 가점
    if any(t in place_types for t in intent.priority_types):
        score += 3.0
    
    # 5. 카테고리별 추가 점수
    if intent.category == "restaurant":
        if "restaurant" in place_types:
            score += 2.0
        if any(kw in place_name for kw in ["맛집", "식당", "레스토랑", "음식점", "카페", "브런치"]):
            score += 1.0
    
    elif intent.category == "cafe":
        if "cafe" in place_types:
            score += 2.0
        if "카페" in place_name:
            score += 1.0
    
    elif intent.category == "emotional_space":
        if any(t in place_types for t in ["cafe", "art_gallery", "museum", "book_store"]):
            score += 2.0
        if any(kw in place_name for kw in ["카페", "갤러리", "갤러", "전시", "문화", "스튜디오", "브런치", "와인바"]):
            score += 1.0
    
    elif intent.category == "tourist_attraction":
        if "tourist_attraction" in place_types:
            score += 2.0
    
    return (True, score)


def filter_places_by_intent(
    places: List[Dict[str, Any]], 
    intent: QueryIntent,
    max_distance_km: float
) -> List[Dict[str, Any]]:
    """의도에 맞게 장소 필터링"""
    filtered = []
    
    for place in places:
        # 거리 체크
        distance = place.get("distance_km", 999)
        if distance > max_distance_km:
            continue
        
        # 의도 체크
        should_include, relevance_score = should_include_place(place, intent)
        if not should_include:
            continue
        
        # 관련도 점수 업데이트
        place["relevance_score"] = relevance_score
        filtered.append(place)
    
    return filtered

