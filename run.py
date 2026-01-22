import os
import json
import math
from typing import Any, Dict, List, Optional, Tuple

import requests
from rich import print
from rich.table import Table

from google import genai
from google.genai import types

from dotenv import load_dotenv
load_dotenv()

# ----------------------------
# 설정
# ----------------------------
PLACES_NEARBY_URL = "https://places.googleapis.com/v1/places:searchNearby"
FIELD_MASK = "places.id,places.displayName,places.formattedAddress,places.location,places.types,places.rating,places.userRatingCount"

DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
DEFAULT_RADIUS_M = int(os.environ.get("DEFAULT_RADIUS_M", "2000"))
DEFAULT_K = int(os.environ.get("DEFAULT_K", "5"))
DEFAULT_MAX_CANDIDATES = int(os.environ.get("DEFAULT_MAX_CANDIDATES", "25"))


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
        print("[yellow]lat/lon이 뒤바뀐 것으로 보여 자동으로 swap 교정합니다.[/yellow]")
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
    max_results: int = 25,
) -> Dict[str, Any]:
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

    resp = client.models.generate_content(
        model=DEFAULT_MODEL,
        contents=[types.Content(role="user", parts=[types.Part(text=json.dumps(instruction, ensure_ascii=False))])],
        config={"response_mime_type": "application/json", "response_json_schema": schema},
    )
    return json.loads(resp.text)


def pretty_print(prompt: str, lat: float, lon: float, radius_m: int, recs: List[Dict[str, Any]]):
    table = Table(title="추천 결과 (Gemini + Places Nearby)")
    table.add_column("Rank", justify="right")
    table.add_column("이름", overflow="fold")
    table.add_column("거리(km)", justify="right")
    table.add_column("주소", overflow="fold")
    table.add_column("이유", overflow="fold")

    for r in recs:
        table.add_row(str(r["rank"]), r["name"], f'{r["distance_km"]:.2f}', r.get("address") or "", r["reason"])

    print(f"[bold]현재 위치[/bold]: lat={lat}, lon={lon} / 반경={radius_m}m")
    print(f"[bold]프롬프트[/bold]: {prompt}\n")
    print(table)


def main():
    print("[bold cyan]근방 여행지 추천 CLI[/bold cyan]\n")

    # 1) 위치 자동/수동
    mode = input("위치 자동 파악(IP 기반) 사용? (Y/n): ").strip().lower()
    if mode in ("", "y", "yes"):
        try:
            lat, lon, meta = get_location_by_ip()
            lat, lon = validate_or_fix_latlon(lat, lon)
            print("[green]자동 위치 파악 성공[/green]")
            print(f"lat={lat}, lon={lon}")
            print(f"meta={meta}\n")
        except Exception as e:
            print(f"[yellow]자동 위치 파악 실패[/yellow]: {e}")
            lat = float(input("현재 위도(lat) 입력 (예: 37.5665): ").strip())
            lon = float(input("현재 경도(lon) 입력 (예: 126.9780): ").strip())
            lat, lon = validate_or_fix_latlon(lat, lon)
    else:
        lat = float(input("현재 위도(lat) 입력 (예: 37.5665): ").strip())
        lon = float(input("현재 경도(lon) 입력 (예: 126.9780): ").strip())
        lat, lon = validate_or_fix_latlon(lat, lon)

    # 2) 프롬프트
    prompt = input("\n원하는 조건/취향 프롬프트 입력: ").strip()
    if not prompt:
        raise SystemExit("프롬프트가 비어있습니다.")

    radius_m = input(f"검색 반경(m) (기본 {DEFAULT_RADIUS_M}): ").strip()
    k = input(f"추천 개수 K (기본 {DEFAULT_K}): ").strip()
    max_candidates = input(f"후보 개수 상한 (기본 {DEFAULT_MAX_CANDIDATES}): ").strip()

    radius_m = int(radius_m) if radius_m else DEFAULT_RADIUS_M
    k = int(k) if k else DEFAULT_K
    max_candidates = int(max_candidates) if max_candidates else DEFAULT_MAX_CANDIDATES

    # 3) Gemini client
    gemini_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_key:
        raise SystemExit("GEMINI_API_KEY 환경변수가 필요합니다.")
    client = genai.Client(api_key=gemini_key)

    # 4) 프롬프트 -> 타입 추출
    included_types = gemini_extract_types(client, prompt)
    if not included_types:
        included_types = ["tourist_attraction"]

    print("\n[bold]검색 타입(included_types)[/bold]")
    print(included_types)

    # 5) Places 후보 수집
    raw = places_nearby_search(
        lat=lat,
        lon=lon,
        radius_m=radius_m,
        included_types=included_types,
        max_results=max_candidates,
    )

    places = raw.get("places", [])
    if not places:
        raise SystemExit("반경 내 후보 장소가 없습니다. 반경(radius_m)을 늘리거나 타입을 바꿔보세요.")

    # 6) 후보 정리 + 거리
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

    # 7) Gemini 재랭킹
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
        raise SystemExit("Gemini가 후보 내에서 추천을 구성하지 못했습니다. 프롬프트/반경/타입을 조정해보세요.")

    # 8) 출력 + 저장
    pretty_print(prompt, lat, lon, radius_m, output)

    save = {
        "user_location": {"lat": lat, "lon": lon},
        "radius_m": radius_m,
        "prompt": prompt,
        "included_types": included_types,
        "recommendations": output,
    }
    with open("result.json", "w", encoding="utf-8") as f:
        json.dump(save, f, ensure_ascii=False, indent=2)

    print("\n[green]result.json 저장 완료[/green]")


if __name__ == "__main__":
    main()