# 웹 인터페이스 사용 가이드

이 프로젝트를 HTML 브라우저에서 간단하게 사용할 수 있도록 웹 인터페이스를 제공합니다.

## 설치 및 실행

### 방법 1: 실행 스크립트 사용 (추천) ⭐

가장 간단한 방법입니다. 실행 스크립트가 자동으로 가상환경을 생성하고 의존성을 설치합니다.

#### macOS / Linux:
```bash
./run_server.sh
```

또는 Python 스크립트로:
```bash
python3 run_server.py
```

#### Windows:
```cmd
run_server.bat
```

또는 Python 스크립트로:
```cmd
python run_server.py
```

### 방법 2: 수동 실행

#### 1. 환경 변수 설정

프로젝트 루트 디렉토리에 `.env` 파일을 생성하고 다음 환경 변수를 설정하세요:

```bash
# 필수: Gemini API 키
GEMINI_API_KEY=your_gemini_api_key_here

# 필수: Google Maps API 키 (지도 표시용)
GOOGLE_MAPS_API_KEY=your_google_maps_api_key_here

# 선택 사항: 기본 설정
GEMINI_MODEL=gemini-2.0-flash
DEFAULT_RADIUS_M=2000
DEFAULT_K=5
DEFAULT_MAX_CANDIDATES=25
```

**중요:** 
- Google Maps API 키는 [Google Cloud Console](https://console.cloud.google.com/)에서 발급받을 수 있습니다.
- Maps JavaScript API를 활성화해야 합니다.
- API 키는 서버에서만 사용되며, 클라이언트에는 안전하게 전달됩니다.

#### 2. 의존성 설치

```bash
pip install -r requirements.txt
```

#### 3. 서버 실행

```bash
python app.py
```

### 접속

서버가 실행되면 브라우저에서 터미널에 표시된 주소로 접속하세요:

```
http://localhost:8080
```

**참고:** 
- 기본 포트는 8080입니다 (5000번 포트는 macOS의 AirPlay Receiver가 사용할 수 있습니다)
- 포트가 사용 중이면 자동으로 다른 포트를 찾아 사용합니다
- 특정 포트를 사용하려면 `.env` 파일에 `PORT=원하는포트번호` 추가

## 사용 방법

1. **위치 설정**
   - "자동 위치 파악 (IP 기반)" 버튼: IP 주소를 기반으로 자동으로 위치를 파악합니다
   - "GPS 위치 사용" 버튼: 브라우저의 GPS 기능을 사용하여 현재 위치를 가져옵니다
   - 수동 입력: 위도와 경도를 직접 입력할 수 있습니다

2. **검색 조건 입력**
   - 원하는 조건/취향을 프롬프트로 입력합니다 (예: "조용한 카페", "예술 작품이 있는 박물관")
   - 검색 반경, 추천 개수, 후보 개수 상한을 설정할 수 있습니다

3. **추천 받기**
   - "추천 받기" 버튼을 클릭하면 AI가 최적의 장소를 추천합니다
   - 결과는 목록과 지도로 표시됩니다

## 기능

- 🌍 IP 기반 자동 위치 파악
- 📍 GPS 위치 사용
- 🤖 Gemini AI를 활용한 맞춤형 장소 추천
- 🗺️ Google Maps를 통한 지도 표시
- 📊 거리, 평점, 리뷰 수 등 상세 정보 제공

## API 엔드포인트

### GET /api/location/auto
IP 기반 자동 위치 파악

### POST /api/recommend
장소 추천 요청

**요청 본문:**
```json
{
  "prompt": "조용한 카페",
  "lat": 37.5665,
  "lon": 126.9780,
  "radius_m": 2000,
  "k": 5,
  "max_candidates": 25
}
```

**응답:**
```json
{
  "success": true,
  "included_types": ["cafe"],
  "recommendations": [
    {
      "rank": 1,
      "name": "장소명",
      "address": "주소",
      "lat": 37.5665,
      "lon": 126.9780,
      "distance_km": 0.5,
      "reason": "추천 이유",
      "rating": 4.5,
      "user_ratings_total": 100
    }
  ]
}
```

